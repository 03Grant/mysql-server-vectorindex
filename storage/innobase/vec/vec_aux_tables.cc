#include "vec_aux_tables.h"
#include "univ.i"
#include "dict0dict.h"      // dict_table_t, dict_index_t, table_id_t
#include "dict0dd.h"         // dd_table_open_on_name_in_mem, dd_create_vec_index_table ...
#include "dict0mem.h"       // dict_mem_table_create/add_col, dict_mem_index_create ...
#include "row0mysql.h"      // row_create_table_for_mysql, row_create_index_for_mysql
#include "mem0mem.h"        // mem_heap_create/free
#include "trx0trx.h"        // trx_set_dict_operation
#include "ut0ut.h"          // ib::info, ib::warn
#include "my_sys.h"         // DEBUG_SYNC_C
#include "my_dbug.h"        // DBUG_EXECUTE_IF / DBUG_SUICIDE
#include "data0type.h"      // dtype_t helpers
#include "ha_innodb.h"      // thd_to_trx
#include "my_bitmap.h"      // bitmap_set_all
#include "vec_index_runtime.h"
#include "vec_meta.h"
#include "current_thd.h"    // current_thd
#include "sql/sql_table.h"  // mysql_rename_table
#include "sql/sql_base.h"   // tdc_remove_table
#include "sql/mysqld.h"     // innodb_hton
#include "sql/dd/cache/dictionary_client.h"  // Dictionary_client::Auto_releaser
#include "sql/key.h"        // key_copy
#include "sql/dd/dd_schema.h"  // Schema_MDL_locker
#include "sql/dd/dictionary.h"  // acquire_exclusive_table_mdl
#include "sql/sql_class.h"  // THD, dd_client()
#include "sql/mdl.h"        // MDL_request
#include "sql/sql_lex.h"    // lex_start / lex_end
#include "sql/dd/string_type.h"  // dd::String_type


#include <cstring>
#include <cstdint>
#include <cctype>
#include <string>
#include <sstream>
#include <vector>
#include <utility>
#include <cstdio>   // std::snprintf
#include <cstddef>  // size_t
#include <limits>
#include <algorithm>
#include <cstdlib>  // strtoull
#include <memory>
#include <unordered_set>
#include <fcntl.h>  // F_WRLCK / F_UNLCK
#include <mutex>
#include <sys/stat.h>
#include <unistd.h>

#include "fts0priv.h"         // fts_parse_sql / fts_eval_sql
#include "fts0fts.h"          // aux_name_vec_t
#include "os0file.h"
#include "pars0pars.h"        // pars_info_* helpers
#include "que0que.h"         // que_graph_free

static dict_table_t* vec_create_one_index_table_pk_compatible(
    trx_t* trx, const dict_index_t* vec_index, const char* base_table_name,
    table_id_t table_id, const std::string& aux_name);

static std::string vec_make_aux_name(const char* parent, const std::string& suffix) {
  if (parent == nullptr) {
    return {};
  }

  const char* slash = std::strchr(parent, '/');
  if (slash == nullptr || slash == parent) {
    return {};
  }

  std::string db(parent, static_cast<size_t>(slash - parent));
  if (db.empty()) {
    return {};
  }

  std::string full;
  full.reserve(db.size() + 1 + suffix.size());
  full.append(db).push_back('/');
  full.append(suffix);
  return full;
}

static std::string vec_aux_full_name(const dict_index_t* index) {
  ut_ad(index && index->table && index->table->name.m_name);

  std::string suffix;
  suffix.reserve(32);
  suffix.append("I_VEC_")
        .append(std::to_string(static_cast<unsigned long long>(index->table->id)))
        .append("_")
        .append(std::to_string(static_cast<unsigned long long>(index->id)));

  return vec_make_aux_name(index->table->name.m_name, suffix);
}

std::string vec_aux_prefix(const dict_index_t *index) {
  return vec_aux_full_name(index);
}

std::string vec_aux_segment_name(const std::string& prefix, uint32_t seg_id) {
  std::string name = prefix;
  name.append("_SEG_").append(std::to_string(static_cast<unsigned long long>(seg_id)));
  return name;
}

std::string vec_aux_mem_name(const std::string& prefix) {
  std::string name = prefix;
  name.append("_MEM");
  return name;
}

bool vec_aux_extract_seg_id(const std::string& full_name,
                            const std::string& prefix,
                            uint32_t* seg_id_out) {
  if (seg_id_out == nullptr) return false;
  const std::string suffix = "_SEG_";
  if (full_name.size() <= prefix.size() + suffix.size()) return false;
  if (full_name.compare(0, prefix.size(), prefix) != 0) return false;
  if (full_name.compare(prefix.size(), suffix.size(), suffix) != 0) return false;
  const std::string id_part = full_name.substr(prefix.size() + suffix.size());
  if (id_part.empty()) return false;
  char* endptr = nullptr;
  unsigned long long val = std::strtoull(id_part.c_str(), &endptr, 10);
  if (endptr == id_part.c_str() || *endptr != '\0') return false;
  *seg_id_out = static_cast<uint32_t>(val);
  return true;
}

bool vec_aux_table_exists(const std::string& full_name) {
  if (full_name.empty()) return false;
  THD *thd = current_thd;
  MDL_ticket *mdl = nullptr;
  #ifdef UNIV_DEBUG
  const bool dict_locked = dict_sys_mutex_own();
#else
  const bool dict_locked = false;
#endif
  dict_table_t *t =
      dd_table_open_on_name_in_mem(full_name.c_str(), dict_locked);
  if (t == nullptr && thd != nullptr && !dict_locked) {
    t = dd_table_open_on_name(thd, &mdl, full_name.c_str(), dict_locked,
                              DICT_ERR_IGNORE_NONE);
  }
  if (t == nullptr) {
    t = dict_table_open_on_name(full_name.c_str(), dict_locked, false,
                                DICT_ERR_IGNORE_NONE);
  }
  if (t != nullptr) {
    dd_table_close(t, thd, &mdl, false);
    return true;
  }
  return false;
}

bool vec_is_aux_table_name(const char* name) {
  if (name == nullptr) {
    return false;
  }

  const char* slash = std::strchr(name, '/');
  if (slash == nullptr || *(slash + 1) == '\0') {
    return false;
  }

  const char* base = slash + 1;
  constexpr const char* prefix = "I_VEC_";
  constexpr size_t prefix_len = 6;

  if (std::strncmp(base, prefix, prefix_len) != 0) {
    return false;
  }

  const char* p = base + prefix_len;
  if (!std::isdigit(static_cast<unsigned char>(*p))) {
    return false;
  }
  while (std::isdigit(static_cast<unsigned char>(*p))) {
    ++p;
  }
  if (*p != '_') {
    return false;
  }
  ++p;
  if (!std::isdigit(static_cast<unsigned char>(*p))) {
    return false;
  }
  while (std::isdigit(static_cast<unsigned char>(*p))) {
    ++p;
  }

  if (*p == '\0') {
    return true;
  }
  if (std::strcmp(p, "_MEM") == 0 || std::strcmp(p, "_NEXT") == 0) {
    return true;
  }
  if (std::strncmp(p, "_SEG_", 5) == 0) {
    p += 5;
    if (!std::isdigit(static_cast<unsigned char>(*p))) {
      return false;
    }
    while (std::isdigit(static_cast<unsigned char>(*p))) {
      ++p;
    }
    return *p == '\0';
  }

  return false;
}

bool vec_dict_table_is_aux(const dict_table_t* table) {
  if (table == nullptr || table->name.m_name == nullptr) {
    return false;
  }
  return vec_is_aux_table_name(table->name.m_name);
}

uint32_t vec_aux_scan_max_segment(const std::string& prefix,
                                  uint32_t probe_limit) {
  uint32_t max_id = 0;
  uint32_t found = 0;
  const uint32_t limit = std::max<uint32_t>(probe_limit, 1);

  for (uint32_t seg = 1; seg <= limit; ++seg) {
    const std::string name = vec_aux_segment_name(prefix, seg);
    if (vec_aux_table_exists(name)) {
      max_id = std::max(max_id, seg);
      ++found;
    }
  }

  // If none found, return 0 to indicate "start from 1".
  return max_id;
}

static std::string vec_aux_suffix_from_full(const std::string& full_name) {
  const auto slash = full_name.find('/');
  if (slash == std::string::npos || slash + 1 >= full_name.size()) {
    return {};
  }
  return full_name.substr(slash + 1);
}

std::string vec_aux_active_name(const dict_index_t* index) {
  if (index != nullptr && index->vec_runtime != nullptr) {
    vec_index_ctx_t* ctx = index->vec_runtime;
    // unique lock?
    std::shared_lock<std::shared_mutex> lk(ctx->mu);
    if (auto* seg = ctx->mutable_segment(); seg != nullptr) {
      if (!seg->aux_table_name.empty()) {
        return seg->aux_table_name;
      }
    }
  }
  return vec_aux_mem_name(vec_aux_full_name(index));
}

/** Update DD metadata for a renamed VEC auxiliary table.
@param[in]      old_name  original fully qualified name
@param[in]      new_name  new fully qualified name
@param[in]      table     dict table object after rename
@return true on success */
static bool vec_aux_update_dd_after_rename(const std::string& old_name,
                                           const std::string& new_name,
                                           const dict_table_t* table) {
  if (table == nullptr || old_name.empty() || new_name.empty()) {
    return false;
  }

  THD* thd = current_thd;
  if (thd == nullptr) {
    return false;
  }

  const ulonglong saved_options = thd->variables.option_bits;
  const bool adjust_autocommit =
      (saved_options & OPTION_AUTOCOMMIT) ||
      !(saved_options & OPTION_NOT_AUTOCOMMIT);
  if (adjust_autocommit) {
    thd->variables.option_bits &= ~OPTION_AUTOCOMMIT;
    thd->variables.option_bits |= OPTION_NOT_AUTOCOMMIT;
  }

  std::string old_db, old_tbl;
  std::string new_db, new_tbl;
  dict_name::get_table(old_name.c_str(), old_db, old_tbl);
  dict_name::get_table(new_name.c_str(), new_db, new_tbl);

  if (old_db.empty() || old_tbl.empty() || new_db.empty() || new_tbl.empty()) {
    if (adjust_autocommit) {
      thd->variables.option_bits = saved_options;
    }
    return false;
  }

  dd::Schema_MDL_locker mdl_locker(thd);
  dd::cache::Dictionary_client* client = dd::get_dd_client(thd);
  dd::cache::Dictionary_client::Auto_releaser releaser(client);

  if (mdl_locker.ensure_locked(new_db.c_str())) {
    if (adjust_autocommit) {
      thd->variables.option_bits = saved_options;
    }
    return false;
  }

  MDL_ticket* mdl_old = nullptr;
  if (dd::acquire_exclusive_table_mdl(thd, old_db.c_str(), old_tbl.c_str(),
                                      false, &mdl_old)) {
    if (adjust_autocommit) {
      thd->variables.option_bits = saved_options;
    }
    return false;
  }

  MDL_ticket* mdl_new = nullptr;
  if (dd::acquire_exclusive_table_mdl(thd, new_db.c_str(), new_tbl.c_str(),
                                      false, &mdl_new)) {
    if (adjust_autocommit) {
      thd->variables.option_bits = saved_options;
    }
    return false;
  }

  dd::Table* dd_table = nullptr;
  if (client->acquire_for_modification<dd::Table>(old_db.c_str(),
                                                  old_tbl.c_str(),
                                                  &dd_table) ||
      dd_table == nullptr) {
    if (adjust_autocommit) {
      thd->variables.option_bits = saved_options;
    }
    return false;
  }

  const dd::Schema* schema = nullptr;
  if (client->acquire<dd::Schema>(new_db.c_str(), &schema) ||
      schema == nullptr) {
    if (adjust_autocommit) {
      thd->variables.option_bits = saved_options;
    }
    return false;
  }

  dd_table->set_schema_id(schema->id());
  dd_table->set_name(new_tbl.c_str());

  if (dict_table_is_file_per_table(table)) {
    char* new_path = fil_space_get_first_path(table->space);
    dberr_t err =
        dd_tablespace_rename(table->dd_space_id, false, new_name.c_str(),
                             new_path);
    if (new_path != nullptr) {
      ut::free(new_path);
    }
    if (err != DB_SUCCESS) {
      if (adjust_autocommit) {
        thd->variables.option_bits = saved_options;
      }
      return false;
    }
  }

  if (client->update(dd_table)) {
    if (adjust_autocommit) {
      thd->variables.option_bits = saved_options;
    }
    return false;
  }

  // Ensure DD client registries are flushed so Auto_releaser destructor
  // sees no pending uncommitted objects in non-transactional DDL path.
  client->commit_modified_objects();

  // Invalidate table definition cache entries for both names so SQL layer
  // doesn't reuse stale TABLE_SHARE objects after internal rename.
  tdc_remove_table(thd, TDC_RT_REMOVE_UNUSED, old_db.c_str(), old_tbl.c_str(),
                   false);
  tdc_remove_table(thd, TDC_RT_REMOVE_UNUSED, new_db.c_str(), new_tbl.c_str(),
                   false);

  if (adjust_autocommit) {
    thd->variables.option_bits = saved_options;
  }
  return true;
}

dberr_t vec_aux_rename_table(trx_t* trx,
                             const std::string& old_name,
                             const std::string& new_name) {
  if (trx == nullptr || old_name.empty() || new_name.empty()) {
    return DB_ERROR;
  }
  if (old_name == new_name) {
    return DB_SUCCESS;
  }

  trx_set_dict_operation(trx, TRX_DICT_OP_TABLE);

  row_mysql_lock_data_dictionary(trx, UT_LOCATION_HERE);
  dberr_t err = row_rename_table_for_mysql(old_name.c_str(), new_name.c_str(),
                                           nullptr, trx, false);
  row_mysql_unlock_data_dictionary(trx);

  if (err != DB_SUCCESS) {
    return err;
  }

  dict_table_t* table =
      dd_table_open_on_name_in_mem(new_name.c_str(), false /*dict_locked*/);
  if (table == nullptr) {
    return DB_ERROR;
  }

  bool dd_ok = vec_aux_update_dd_after_rename(old_name, new_name, table);
  dd_table_close(table, current_thd, nullptr, false);

  return dd_ok ? DB_SUCCESS : DB_ERROR;
}

dberr_t vec_aux_create_table(trx_t* trx,
                             dict_index_t* index,
                             const std::string& full_name) {
  if (trx == nullptr || index == nullptr || full_name.empty()) {
    return DB_ERROR;
  }
  if (index->table == nullptr || index->table->name.m_name == nullptr) {
    return DB_ERROR;
  }

  const std::string suffix = vec_aux_suffix_from_full(full_name);
  if (suffix.empty()) {
    return DB_ERROR;
  }

  dict_table_t* t = vec_create_one_index_table_pk_compatible(
      trx, index, index->table->name.m_name, index->table->id, suffix);
  return t != nullptr ? DB_SUCCESS : DB_ERROR;
}

static inline uint64_t rowid6_to_u64(const byte rowid6[6]) {
  // InnoDB storage 用 big-endian。手动拼或先放入8字节buffer高位为0。
  uint64_t v = 0;
  v |= (uint64_t)rowid6[0] << 40;
  v |= (uint64_t)rowid6[1] << 32;
  v |= (uint64_t)rowid6[2] << 24;
  v |= (uint64_t)rowid6[3] << 16;
  v |= (uint64_t)rowid6[4] << 8;
  v |= (uint64_t)rowid6[5];
  return v;
}

// 把 u64 写成 InnoDB storage 序的 8 字节（用于绑定 BIGINT）
static inline void u64_to_storage8(uint64_t v, byte out[8]) {
  mach_write_to_8(out, (ulonglong)v); // InnoDB自带：写 big-endian 8B
}


static bool is_gen_clust_name(const dict_index_t* idx) {
  if (!idx || !idx->name) return false;
  return std::strcmp(idx->name, "GEN_CLUST_INDEX") == 0;
}

struct vec_aux_pk_meta_t {
  dict_index_t* clust{nullptr};
  bool use_row_id{false};
  ulint pk_cols{0};
};

static dberr_t vec_aux_prepare_pk_meta(dict_index_t* index,
                                       vec_aux_pk_meta_t* meta) {
  if (index == nullptr || meta == nullptr) {
    return DB_ERROR;
  }

  dict_table_t* base = index->table;
  if (base == nullptr) {
    ib::warn() << "VECINDEX: index has no base table";
    return DB_ERROR;
  }

  dict_index_t* clust = base->first_index();
  if (clust == nullptr) {
    ib::warn() << "VECINDEX: base table has no clustered index";
    return DB_ERROR;
  }

  meta->clust = clust;
  meta->use_row_id = is_gen_clust_name(clust);
  meta->pk_cols = meta->use_row_id ? 1 : clust->n_uniq;

  return DB_SUCCESS;
}

static dberr_t vec_aux_bind_pk_column_ids(pars_info_t* info,
                                          const vec_aux_pk_meta_t& meta) {
  if (info == nullptr || meta.clust == nullptr) {
    return DB_ERROR;
  }

  for (ulint i = 0; i < meta.pk_cols; ++i) {
    const char* colname = nullptr;
    if (meta.use_row_id) {
      colname = "row_id";
    } else {
      dict_field_t* field = meta.clust->get_field(i);
      if (field == nullptr || field->name == nullptr) {
        ib::warn() << "VECINDEX: failed to fetch PK column meta at position "
                   << i;
        return DB_ERROR;
      }
      colname = static_cast<const char*>(field->name);
    }

    char key[8];
    snprintf(key, sizeof(key), "c%u", static_cast<unsigned>(i));
    pars_info_bind_id(info, true, key, colname);
  }

  char key[8];
  snprintf(key, sizeof(key), "c%u", static_cast<unsigned>(meta.pk_cols));
  pars_info_bind_id(info, true, key, "faiss_id");

  return DB_SUCCESS;
}

static const char* vec_aux_literal_name(pars_info_t* info, ulint idx) {
  ut_ad(info != nullptr && info->heap != nullptr);

  char key[16];
  snprintf(key, sizeof(key), "v%u", static_cast<unsigned>(idx));

  if (auto* existing = pars_info_get_bound_lit(info, key); existing != nullptr) {
    return existing->name;
  }

  return mem_heap_strdup(info->heap, key);
}

static dberr_t vec_aux_bind_pk_values(
    pars_info_t* info, const vec_aux_pk_meta_t& meta,
    const std::vector<vec_pk_column_t>& pk_columns) {
  if (info == nullptr) {
    return DB_ERROR;
  }

  if (meta.use_row_id) {
    if (pk_columns.empty() || pk_columns[0].is_null) {
      ib::warn() << "VECINDEX: NULL in PK column for row_id not allowed";
      return DB_ERROR;
    }
    const auto& c = pk_columns[0];
    if (c.data.size() != 6) {
      ib::warn() << "Row_id snapshot is not 6 bytes.";
      return DB_ERROR;
    }

    const char* vname = vec_aux_literal_name(info, 0);
    byte* be8 = static_cast<byte*>(mem_heap_alloc(info->heap, 8));
    uint64_t v = rowid6_to_u64(reinterpret_cast<const byte*>(c.data.data()));
    u64_to_storage8(v, be8);
    pars_info_bind_literal(info, vname, be8, sizeof(be8), DATA_INT,
                           DATA_UNSIGNED);
    return DB_SUCCESS;
  }

  if (pk_columns.size() < meta.pk_cols) {
    ib::warn() << "VECINDEX: PK column snapshot missing entries, have "
               << pk_columns.size() << " expect " << meta.pk_cols;
    return DB_ERROR;
  }

  for (ulint i = 0; i < meta.pk_cols; ++i) {
    const auto& col = pk_columns[i];

    const char* vname = vec_aux_literal_name(info, i);

    const void* ptr = (col.is_null || col.data.empty())
                          ? nullptr
                          : static_cast<const void*>(col.data.data());
    assert(ptr != nullptr);
    uint32_t len = static_cast<uint32_t>(col.data.size());

    pars_info_bind_literal(info, vname, ptr, len, col.mtype, col.prtype);
  }

  return DB_SUCCESS;
}

static bool vec_extract_pk_from_aux(TABLE* mysql_table,
                                    dict_index_t* clust_index,
                                    ulint pk_fields,
                                    std::vector<vec_pk_column_t>& out) {
  if (mysql_table == nullptr || clust_index == nullptr ||
      mysql_table->s == nullptr || pk_fields == 0) {
    return false;
  }

  out.clear();
  out.resize(pk_fields);

  for (ulint i = 0; i < pk_fields; ++i) {
    if (i >= static_cast<ulint>(mysql_table->s->fields)) {
      return false;
    }
    Field* f = mysql_table->field[i];
    if (f == nullptr) {
      return false;
    }

    vec_pk_column_t col{};
    col.is_null = f->is_null();

    dict_field_t* df = clust_index->get_field(i);
    if (df != nullptr && df->col != nullptr) {
      col.mtype = df->col->mtype;
      col.prtype = df->col->prtype;
    }

    if (!col.is_null) {
      const uchar* ptr = f->data_ptr();
      const uint len = f->data_length();
      if (ptr == nullptr && len != 0) {
        return false;
      }

      dtype_t dtype;
      dtype_set(&dtype, col.mtype, col.prtype, df->col->len);
      dfield_t dfield;
      dfield_set_type(&dfield, &dtype);
      std::vector<byte> tmp(df->col->len + 16);
      byte* end = row_mysql_store_col_in_innobase_format(
          &dfield, tmp.data(), true, ptr, df->col->len,
          dict_table_is_comp(clust_index->table));
      const ulint stored_len =
          static_cast<ulint>(end - static_cast<byte*>(tmp.data()));
      col.data.assign(tmp.data(), tmp.data() + stored_len);
    }

    out[i] = std::move(col);
  }

  return true;
}

dberr_t vec_aux_insert_one(
    trx_t* trx,
    dict_index_t* index,
    const std::vector<vec_pk_column_t>& pk_columns,
    uint64_t faiss_id){
  pars_info_t *info = pars_info_create();
  const std::string aux_full = vec_aux_active_name(index);
  pars_info_bind_id(info, true, "index_table_name", aux_full.c_str());

  vec_aux_pk_meta_t meta;
  dberr_t err = vec_aux_prepare_pk_meta(index, &meta);
  if (err != DB_SUCCESS) {
    pars_info_free(info);
    return err;
  }

  err = vec_aux_bind_pk_column_ids(info, meta);
  if (err != DB_SUCCESS) {
    pars_info_free(info);
    return err;
  }

  err = vec_aux_bind_pk_values(info, meta, pk_columns);
  if (err != DB_SUCCESS) {
    pars_info_free(info);
    return err;
  }

  /* Store FAISS id as unsigned 64-bit. */
  pars_info_add_ull_literal(info, "faiss_id", faiss_id);

  /* InnoDB's internal SQL parser (pars0grm.yy) only supports INSERT ... VALUES
  without a column list, so the aux table definition must keep PK columns first
  followed by faiss_id. */
  std::ostringstream sql;
  sql << "BEGIN\nINSERT INTO $index_table_name VALUES (";
  for (ulint i = 0; i < meta.pk_cols; ++i) {
    if (i != 0) {
      sql << ", ";
    }
    sql << ":v" << i;
  }
  if (meta.pk_cols > 0) {
    sql << ", ";
  }
  sql << ":faiss_id);";

  que_t* graph = vec_parse_sql(aux_full.c_str(), info, sql.str().c_str());

  dberr_t error = vec_eval_sql(trx, graph);
  if(error != DB_SUCCESS){
    ib::warn() << "vec_eval_sql failed with error:" << error;
    que_graph_free(graph);
    return error;
  }

  que_graph_free(graph);



  return DB_SUCCESS;
                          
}


dberr_t vec_aux_insert_pk_null(trx_t* trx,
    dict_index_t* index,
    const std::vector<vec_pk_column_t>& pk_columns){
  ib::warn() << "vec_aux_insert_pk_null called.";
  pars_info_t *info = pars_info_create();
  
  ib::warn() << "vec_aux_insert_pk_null created pars_info.";

  const std::string aux_full = vec_aux_active_name(index);
  pars_info_bind_id(info, true, "index_table_name", aux_full.c_str());
  ib::warn() << "vec_aux_insert_pk_null called for index: " << aux_full;

  vec_aux_pk_meta_t meta;
  dberr_t err = vec_aux_prepare_pk_meta(index, &meta);
  if (err != DB_SUCCESS) {
    pars_info_free(info);
    ib::warn() << "vec_aux_insert_pk_null failed at prepare_pk_meta, error:" << err;
    return err;
  }
  ib::warn() << "vec_aux_insert_pk_null prepare_pk_meta success.";

  err = vec_aux_bind_pk_column_ids(info, meta);
  if (err != DB_SUCCESS) {
    pars_info_free(info);
    ib::warn() << "vec_aux_insert_pk_null failed at bind_pk_column_ids, error:" << err;
    return err;
  }
  ib::warn() << "vec_aux_insert_pk_null bind_pk_column_ids success.";

  err = vec_aux_bind_pk_values(info, meta, pk_columns);
  if (err != DB_SUCCESS) {
    pars_info_free(info);
    ib::warn() << "vec_aux_insert_pk_null failed at bind_pk_values, error:" << err;
    return err;
  }
  ib::warn() << "vec_aux_insert_pk_null bind_pk_values success.";

  /* Insert sentinel UINT64_MAX instead of NULL because the aux table column is
  NONNULL and unsigned. */
  const char *faiss_null_name = "faiss_id_null";
  pars_info_add_ull_literal(info, faiss_null_name, UINT64_MAX);

  /* Same parser restriction as vec_aux_insert_one(): rely on table column
  order for the INSERT target list. */
  std::ostringstream sql;
  sql << "BEGIN\nINSERT INTO $index_table_name VALUES (";
  for (ulint i = 0; i < meta.pk_cols; ++i) {
    if (i != 0) {
      sql << ", ";
    }
    sql << ":v" << i;
  }
  if (meta.pk_cols > 0) {
    sql << ", ";
  }
  sql << ":" << faiss_null_name << ");";
  ib::warn() << "vec_aux_insert_pk_null executing SQL: " << sql.str();
  que_t* graph = vec_parse_sql(aux_full.c_str(), info, sql.str().c_str());
  ib::warn() << "vec_aux_insert_pk_null parsed SQL into graph.";

  dberr_t error = vec_eval_sql(trx, graph);
  ib::warn() << "vec_aux_insert_pk_null vec_eval_sql returned: " << error;
  if (error != DB_SUCCESS) {
    ib::warn() << "vec_aux_insert_pk_null failed with error:" << error;
    if (trx->error_state == DB_SUCCESS) {
      trx->error_state = error;
    }
    que_graph_free(graph);
    return error;
  }

  que_graph_free(graph);

  /* Debug-only injection: pause/crash right after inserting the sentinel row,
  before the caller continues (used to simulate crash during forward roll). */
  DEBUG_SYNC_C("vec_aux_insert_pk_null_after");
  DBUG_EXECUTE_IF("crash_vec_aux_insert_pk_null_after", DBUG_SUICIDE(););


  return DB_SUCCESS;
}

dberr_t vec_aux_update_pk_vid(trx_t* trx,
    dict_index_t* index,
    const std::vector<vec_pk_column_t>& pk_columns,
    uint64_t faiss_id){
  
  DEBUG_SYNC_C("vec_aux_update_pk_vid_after");
  DBUG_EXECUTE_IF("crash_vec_aux_update_pk_vid_after", DBUG_SUICIDE(););


  ib::warn() << "vec_aux_update_pk_vid called.";
  pars_info_t *info = pars_info_create();
  const std::string aux_full = vec_aux_active_name(index);
  pars_info_bind_id(info, true, "index_table_name", aux_full.c_str());

  vec_aux_pk_meta_t meta;
  dberr_t err = vec_aux_prepare_pk_meta(index, &meta);
  if (err != DB_SUCCESS) {
    pars_info_free(info);
    return err;
  }

  err = vec_aux_bind_pk_column_ids(info, meta);
  if (err != DB_SUCCESS) {
    pars_info_free(info);
    return err;
  }

  err = vec_aux_bind_pk_values(info, meta, pk_columns);
  if (err != DB_SUCCESS) {
    pars_info_free(info);
    return err;
  }

  pars_info_add_ull_literal(info, "faiss_id", faiss_id);

  ib::warn() << "vec_aux_update_pk_vid preparing SQL statement.";

  std::ostringstream sql;
  sql << "BEGIN\nUPDATE $index_table_name SET $c" << meta.pk_cols
      << " = :faiss_id WHERE ";
  for (ulint i = 0; i < meta.pk_cols; ++i) {
    if (i != 0) {
      sql << " AND ";
    }
    sql << "$c" << i << " = :v" << i;
  }
  sql << ";";

  ib::warn() << "vec_aux_update_pk_vid executing SQL: " << sql.str();
  que_t* graph = vec_parse_sql(aux_full.c_str(), info, sql.str().c_str());
  
  ib::warn() << "vec_aux_update_pk_vid parsed SQL into graph.";
  dberr_t error = vec_eval_sql(trx, graph);
  if (error != DB_SUCCESS) {
    ib::warn() << "vec_aux_update_pk_vid failed with error:" << error;
    if (trx->error_state == DB_SUCCESS) {
      trx->error_state = error;
    }
    que_graph_free(graph);
    return error;
  }

  ib::warn() << "vec_aux_update_pk_vid vec_eval_sql returned: " << error;

  que_graph_free(graph);
  return DB_SUCCESS;
}

/** Write PK columns directly into mysql_table->record[0]. */
static bool vec_apply_pk_columns_to_record(
    const std::vector<vec_pk_column_t>& pk_columns,
    dict_index_t* clust_index, ulint pk_fields, const KEY* pk_info,
    TABLE* mysql_table) {
  if (pk_columns.empty() || mysql_table == nullptr || clust_index == nullptr ||
      pk_info == nullptr || mysql_table->record[0] == nullptr ||
      pk_fields == 0) {
    ib::warn() << "vec_apply_pk_columns_to_record: invalid input"
               << " cols=" << pk_columns.size()
               << " pk_fields=" << pk_fields
               << " table=" << (mysql_table ? mysql_table->s->table_name.str
                                            : "<null>");
    return false;
  }

  if (pk_fields != pk_info->user_defined_key_parts) {
    ib::warn() << "vec_apply_pk_columns_to_record: pk_fields mismatch"
               << " cols=" << pk_fields
               << " aux_pk_parts=" << pk_info->user_defined_key_parts;
    return false;
  }

  if (pk_columns.size() < pk_fields) {
    ib::warn() << "vec_apply_pk_columns_to_record: pk_columns insufficient"
               << " cols=" << pk_columns.size()
               << " expected=" << pk_fields;
    return false;
  }

  ib::warn() << "vec_apply_pk_columns_to_record: PK snapshot "
             << vec_format_pk_columns_debug(pk_columns, pk_fields, 24);
  ib::warn() << "vec_apply_pk_columns_to_record: columns -> record conversion";

  const bool use_row_id = is_gen_clust_name(clust_index);

  for (ulint i = 0; i < pk_fields; ++i) {
    Field* f = pk_info->key_part[i].field;
    if (f == nullptr) {
      ib::warn() << "vec_apply_pk_columns_to_record: missing field"
                 << " i=" << i
                 << " field_null=true";
      return false;
    }

    if (i >= pk_columns.size()) {
      ib::warn() << "vec_apply_pk_columns_to_record: pk_columns out of range"
                 << " i=" << i
                 << " pk_columns_sz=" << pk_columns.size();
      return false;
    }

    const vec_pk_column_t& col = pk_columns[i];

    if (col.is_null) {
      if (!f->is_nullable()) {
        ib::warn() << "vec_apply_pk_columns_to_record: NULL for non-nullable col"
                   << " i=" << i
                   << " colname=" << (f->field_name ? f->field_name : "?");
        return false;
      }
      f->set_null();
      continue;
    }

    f->set_notnull();

    if (use_row_id) {
      if (col.data.size() != DATA_ROW_ID_LEN) {
        ib::warn() << "vec_apply_pk_columns_to_record: row_id length mismatch"
                   << " got=" << col.data.size()
                   << " expect=" << DATA_ROW_ID_LEN;
        return false;
      }
      const uint64_t row_id =
          rowid6_to_u64(reinterpret_cast<const byte*>(col.data.data()));
      if (f->store(static_cast<ulonglong>(row_id), true) != 0) {
        ib::warn() << "vec_apply_pk_columns_to_record: store row_id failed"
                   << " row_id=" << row_id;
        return false;
      }
      continue;
    }

    dict_field_t* df = clust_index->get_field(i);
    if (df == nullptr || df->col == nullptr) {
      ib::warn() << "vec_apply_pk_columns_to_record: missing dict field"
                 << " i=" << i
                 << " df_null=" << (df == nullptr);
      return false;
    }

    const bool is_unsigned = (df->col->prtype & DATA_UNSIGNED) != 0 ||
                             f->is_unsigned();

    switch (df->col->mtype) {
      case DATA_INT: {
        const size_t len = col.data.size();
        if (len == 0 || len > sizeof(uint64_t)) {
          ib::warn() << "vec_apply_pk_columns_to_record: DATA_INT bad length"
                     << " len=" << len;
          return false;
        }

        uint64_t v = 0;
        for (size_t j = 0; j < len; ++j) {
          unsigned char b = col.data[j];
          if (!is_unsigned && j == 0) {
            b ^= 0x80;
          }
          v = (v << 8) | static_cast<uint64_t>(b);
        }

        if (is_unsigned) {
          if (f->store(static_cast<ulonglong>(v), true) != 0) {
            ib::warn() << "vec_apply_pk_columns_to_record: store unsigned int failed"
                       << " v=" << v
                       << " len=" << len;
            return false;
          }
        } else {
          const unsigned shift =
              static_cast<unsigned>((sizeof(uint64_t) - len) * 8);
          const int64_t signed_v =
              static_cast<int64_t>((static_cast<int64_t>(v << shift)) >> shift);
          if (f->store(static_cast<longlong>(signed_v), false) != 0) {
            ib::warn() << "vec_apply_pk_columns_to_record: store signed int failed"
                       << " v=" << signed_v
                       << " len=" << len;
            return false;
          }
        }
        break;
      }
      case DATA_VARCHAR:
      case DATA_VARMYSQL:
      case DATA_CHAR:
      case DATA_BINARY:
      case DATA_FIXBINARY: {
        const CHARSET_INFO* cs = f->charset();
        if (f->store(reinterpret_cast<const char*>(col.data.data()),
                     static_cast<uint>(col.data.size()),
                     cs != nullptr ? cs : &my_charset_bin) != 0) {
          ib::warn() << "vec_apply_pk_columns_to_record: store string/binary failed"
                     << " len=" << col.data.size()
                     << " colname=" << (f->field_name ? f->field_name : "?");
          return false;
        }
        break;
      }
      default:
        ib::warn() << "vec_apply_pk_columns_to_record: unsupported mtype"
                   << " i=" << i
                   << " mtype=" << df->col->mtype;
        return false;
    }
  }

  return true;
}

dberr_t vec_aux_handler_update(trx_t* trx,
                               const std::string& table_name,
                               dict_index_t* clust_index,
                               ulint pk_fields,
                               const std::vector<vec_pk_column_t>& old_pk_columns,
                               const std::vector<vec_pk_column_t>& new_pk_columns,
                               uint64_t* out_vid) {
  ib::warn() << "vec_aux_handler_update called.";
  if (table_name.empty() || clust_index == nullptr || pk_fields == 0 ||
      old_pk_columns.size() < pk_fields || new_pk_columns.size() < pk_fields) {
    ib::warn() << "vec_aux_handler_update: invalid input parameters"
               << " old_pk_columns=" << old_pk_columns.size()
               << " new_pk_columns=" << new_pk_columns.size()
               << " pk_fields=" << pk_fields;
    return DB_RECORD_NOT_FOUND;
  }

  THD* thd = (trx != nullptr && trx->mysql_thd != nullptr)
                 ? trx->mysql_thd
                 : current_thd;
  if (thd == nullptr) {
    ib::warn() << "vec_aux_handler_update: no THD available";
    return DB_ERROR;
  }

  const auto slash = table_name.find('/');
  if (slash == std::string::npos || slash == 0 ||
      slash + 1 >= table_name.size()) {
    ib::warn() << "vec_aux_handler_update: invalid table name format";
    return DB_RECORD_NOT_FOUND;
  }
  const std::string db = table_name.substr(0, slash);
  const std::string tbl = table_name.substr(slash + 1);

  if (trx != nullptr && thd_to_trx(thd) != trx) {
    ib::warn() << "vec_aux_handler_update: aligning THD trx with caller trx";
    thd_to_trx(thd) = trx;
  }

  ib::warn() << "vec_aux_handler_update: acquiring MDL lock for table ";
  MDL_request mdl_request;
  MDL_REQUEST_INIT(&mdl_request, MDL_key::TABLE, db.c_str(), tbl.c_str(),
                   MDL_SHARED_WRITE, MDL_EXPLICIT);
  if (thd->mdl_context.acquire_lock(&mdl_request,
                                    thd->variables.lock_wait_timeout)) {
    ib::warn() << "vec_aux_handler_update: MDL lock acquire failed";
    return DB_LOCK_WAIT_TIMEOUT;
  }

  struct MdlGuard {
    THD* thd{nullptr};
    MDL_request* req{nullptr};
    ~MdlGuard() {
      if (thd != nullptr && req != nullptr && req->ticket != nullptr) {
        thd->mdl_context.release_lock(req->ticket);
        req->ticket = nullptr;
      }
    }
  } mdl_guard{thd, &mdl_request};

  dd::cache::Dictionary_client* client = dd::get_dd_client(thd);
  dd::cache::Dictionary_client::Auto_releaser releaser(client);
  const dd::Table* dd_table_obj = nullptr;
  if (client->acquire<dd::Table>(db.c_str(), tbl.c_str(), &dd_table_obj) ||
      dd_table_obj == nullptr) {
    ib::warn() << "vec_aux_handler_update: failed to acquire DD table object";
    return DB_ERROR;
  }

  ib::warn() << "vec_aux_handler_update: opening table for read";
  char path[FN_REFLEN + 1]{};
  bool truncated = false;
  size_t path_len = build_table_filename(
      path, sizeof(path) - 1 - reg_ext_length, db.c_str(), tbl.c_str(), "", 0,
      &truncated);
  if (path_len == 0 || truncated) {
    return DB_ERROR;
  }
  ib::warn() << "vec_aux_handler_update: opening table uncached";

  TABLE* mysql_table = open_table_uncached(
      thd, path, db.c_str(), tbl.c_str(), false, true, *dd_table_obj);
  if (mysql_table == nullptr || mysql_table->file == nullptr) {
    return DB_RECORD_NOT_FOUND;
  }

  struct TableGuard {
    TABLE* t{nullptr};
    ~TableGuard() {
      if (t != nullptr) {
        intern_close_table(t);
        t = nullptr;
      }
    }
  } table_guard{mysql_table};

  handler* h = mysql_table->file;
  if (mysql_table->read_set != nullptr) {
    bitmap_set_all(mysql_table->read_set);
  }
  if (mysql_table->write_set != nullptr) {
    bitmap_set_all(mysql_table->write_set);
  }

  if (h != nullptr) {
    ha_innobase* innodb = dynamic_cast<ha_innobase*>(h);
    if (innodb != nullptr) {
      innodb->init_table_handle_for_HANDLER();
    }
  }

  trx_t* stmt_trx = thd_to_trx(thd);
  if (stmt_trx == nullptr && trx != nullptr) {
    thd_to_trx(thd) = trx;
    stmt_trx = trx;
  }
  if (stmt_trx == nullptr) {
    return DB_ERROR;
  }

  ib::warn() << "vec_aux_handler_update: starting transaction if not started";
  trx_start_if_not_started(stmt_trx, true, UT_LOCATION_HERE);
  struct TrxDepthGuard {
    trx_t* trx{nullptr};
    bool entered{false};
    ~TrxDepthGuard() {
      if (entered && trx != nullptr) {
        TrxInInnoDB::end_stmt(trx);
      }
    }
  } depth_guard{stmt_trx, false};
  TrxInInnoDB::begin_stmt(stmt_trx);
  depth_guard.entered = true;

  auto map_handler_err = [](int rc) -> dberr_t {
    switch (rc) {
      case 0:
      case HA_ERR_RECORD_IS_THE_SAME:
        return DB_SUCCESS;
      case HA_ERR_LOCK_WAIT_TIMEOUT:
        return DB_LOCK_WAIT_TIMEOUT;
      case HA_ERR_LOCK_DEADLOCK:
        return DB_DEADLOCK;
      case HA_ERR_KEY_NOT_FOUND:
      case HA_ERR_END_OF_FILE:
        return DB_RECORD_NOT_FOUND;
      default:
        return DB_ERROR;
    }
  };

  if (mysql_table->s != nullptr && mysql_table->s->rec_buff_length > 0) {
    std::memset(mysql_table->record[0], 0, mysql_table->s->rec_buff_length);
  }

  /* Acquire engine-level table lock so InnoDB sees LOCK_IX. */
  bool locked = false;
  auto unlock_if_needed = [&](handler *handler_ptr) {
    if (locked && handler_ptr != nullptr) {
      handler_ptr->ha_external_lock(thd, F_UNLCK);
      locked = false;
    }
  };
  int lock_rc = h->ha_external_lock(thd, F_WRLCK);
  dberr_t lock_err = map_handler_err(lock_rc);
  if (lock_err != DB_SUCCESS) {
    unlock_if_needed(h);
    return lock_err;
  }
  locked = true;

  struct LockGuard {
    handler *h;
    THD *thd;
    bool &locked_ref;
    ~LockGuard() {
      if (locked_ref && h != nullptr) {
        h->ha_external_lock(thd, F_UNLCK);
        locked_ref = false;
      }
    }
  } lock_guard{h, thd, locked};

  if (mysql_table->s->primary_key == MAX_KEY) {
    ib::warn() << "vec_aux_handler_update: no primary key on aux table";
    return DB_ERROR;
  }

  KEY* pk_info = mysql_table->key_info + mysql_table->s->primary_key;

  if (pk_fields != static_cast<ulint>(pk_info->user_defined_key_parts)) {
    ib::warn() << "vec_aux_handler_update: pk_fields=" << pk_fields
               << " aux_pk_parts=" << pk_info->user_defined_key_parts;
    return DB_SCHEMA_MISMATCH;
  }

  ib::warn() << "vec_aux_handler_update: old pk columns -> record buffer";
  if (!vec_apply_pk_columns_to_record(old_pk_columns, clust_index, pk_fields,
                                      pk_info, mysql_table)) {
    ib::warn()
        << "vec_aux_handler_update: failed to apply old pk_columns to record";
    return DB_SCHEMA_MISMATCH;
  }

  std::vector<uchar> key_buf(static_cast<size_t>(pk_info->key_length));
  key_copy(key_buf.data(), mysql_table->record[0], pk_info, 0);

  int rc = h->ha_index_read_idx_map(
      mysql_table->record[0], mysql_table->s->primary_key, key_buf.data(),
      make_prev_keypart_map(pk_info->user_defined_key_parts),
      HA_READ_KEY_EXACT);
  dberr_t err = map_handler_err(rc);
  if (err != DB_SUCCESS) {
    ib::warn() << "vec_aux_handler_update: index_read_idx_map failed rc="
               << rc;
    return err;
  }

  Field* faiss_field =
      (pk_fields < static_cast<ulint>(mysql_table->s->fields))
          ? mysql_table->field[pk_fields]
          : nullptr;
  if (faiss_field == nullptr || faiss_field->is_null()) {
    return DB_RECORD_NOT_FOUND;
  }

  const uint64_t vid =
      static_cast<uint64_t>(static_cast<ulonglong>(faiss_field->val_int()));

  if (mysql_table->record[1] == nullptr ||
      mysql_table->s == nullptr) {
    ib::warn() << "vec_aux_handler_update: missing record[1] buffer";
    return DB_ERROR;
  }

  std::memcpy(mysql_table->record[1], mysql_table->record[0],
              mysql_table->s->rec_buff_length);

  ib::warn() << "vec_aux_handler_update: new pk columns -> record buffer";
  if (!vec_apply_pk_columns_to_record(new_pk_columns, clust_index, pk_fields,
                                      pk_info, mysql_table)) {
    ib::warn()
        << "vec_aux_handler_update: failed to apply new pk_columns to record";
    return DB_SCHEMA_MISMATCH;
  }

  rc = h->ha_update_row(mysql_table->record[1], mysql_table->record[0]);
  err = map_handler_err(rc);
  if (err == DB_SUCCESS && out_vid != nullptr) {
    *out_vid = vid;
  }

  return err;
}

dberr_t vec_aux_handler_delete(trx_t* trx,
                               const std::string& table_name,
                               dict_index_t* clust_index,
                               ulint pk_fields,
                               const std::vector<vec_pk_column_t>& pk_columns,
                               uint64_t* out_vid) {

  ib::warn() << "vec_aux_handler_delete called.";
  if (table_name.empty() || clust_index == nullptr || pk_fields == 0 ||
      pk_columns.empty()) {
    ib::warn() << "vec_aux_handler_delete: invalid input parameters"
               << " pk_columns=" << pk_columns.size()
               << " dump=" << vec_format_pk_columns_debug(pk_columns, pk_fields);
    return DB_RECORD_NOT_FOUND;
  }

  THD* thd = (trx != nullptr && trx->mysql_thd != nullptr)
                 ? trx->mysql_thd
                 : current_thd;
  if (thd == nullptr) {
    ib::warn() << "vec_aux_handler_delete: no THD available";
    return DB_ERROR;
  }

  const auto slash = table_name.find('/');
  if (slash == std::string::npos || slash == 0 || slash + 1 >= table_name.size()) {
    ib::warn() << "vec_aux_handler_delete: invalid table name format";
    return DB_RECORD_NOT_FOUND;
  }
  const std::string db = table_name.substr(0, slash);
  const std::string tbl = table_name.substr(slash + 1);

  if (trx != nullptr && thd_to_trx(thd) != trx) {
    ib::warn() << "vec_aux_handler_delete: aligning THD trx with caller trx";
    thd_to_trx(thd) = trx;
  }

  ib::warn() << "vec_aux_handler_delete: acquiring MDL lock for table ";
  MDL_request mdl_request;
  MDL_REQUEST_INIT(&mdl_request, MDL_key::TABLE, db.c_str(), tbl.c_str(),
                   MDL_SHARED_WRITE, MDL_EXPLICIT);
  if (thd->mdl_context.acquire_lock(&mdl_request,
                                    thd->variables.lock_wait_timeout)) {
    ib::warn() << "vec_aux_handler_delete: MDL lock acquire failed";                                  
    return DB_LOCK_WAIT_TIMEOUT;
  }

  struct MdlGuard {
    THD* thd{nullptr};
    MDL_request* req{nullptr};
    ~MdlGuard() {
      if (thd != nullptr && req != nullptr && req->ticket != nullptr) {
        thd->mdl_context.release_lock(req->ticket);
        req->ticket = nullptr;
      }
    }
  } mdl_guard{thd, &mdl_request};

  dd::cache::Dictionary_client* client = dd::get_dd_client(thd);
  dd::cache::Dictionary_client::Auto_releaser releaser(client);
  const dd::Table* dd_table_obj = nullptr;
  if (client->acquire<dd::Table>(db.c_str(), tbl.c_str(), &dd_table_obj) ||
      dd_table_obj == nullptr) {
    ib::warn() << "vec_aux_handler_delete: failed to acquire DD table object";
    return DB_ERROR;
  }

  ib::warn() << "vec_aux_handler_delete: opening table for read";
  char path[FN_REFLEN + 1]{};
  bool truncated = false;
  size_t path_len = build_table_filename(
      path, sizeof(path) - 1 - reg_ext_length, db.c_str(), tbl.c_str(), "", 0,
      &truncated);
  if (path_len == 0 || truncated) {
    return DB_ERROR;
  }
  ib::warn() << "vec_aux_handler_delete: opening table uncached";

  TABLE* mysql_table = open_table_uncached(
      thd, path, db.c_str(), tbl.c_str(), false, true, *dd_table_obj);
  if (mysql_table == nullptr || mysql_table->file == nullptr) {
    return DB_RECORD_NOT_FOUND;
  }

  struct TableGuard {
    TABLE* t{nullptr};
    ~TableGuard() {
      if (t != nullptr) {
        intern_close_table(t);
        t = nullptr;
      }
    }
  } table_guard{mysql_table};

  handler* h = mysql_table->file;
  if (mysql_table->read_set != nullptr) {
    bitmap_set_all(mysql_table->read_set);
  }
  if (mysql_table->write_set != nullptr) {
    bitmap_set_all(mysql_table->write_set);
  }

  if (h != nullptr) {
    ha_innobase* innodb = dynamic_cast<ha_innobase*>(h);
    if (innodb != nullptr) {
      innodb->init_table_handle_for_HANDLER();
    }
  }

  trx_t* stmt_trx = thd_to_trx(thd);
  if (stmt_trx == nullptr && trx != nullptr) {
    thd_to_trx(thd) = trx;
    stmt_trx = trx;
  }
  if (stmt_trx == nullptr) {
    return DB_ERROR;
  }

  ib::warn() << "vec_aux_handler_delete: starting transaction if not started";
  trx_start_if_not_started(stmt_trx, true, UT_LOCATION_HERE);
  struct TrxDepthGuard {
    trx_t* trx{nullptr};
    bool entered{false};
    ~TrxDepthGuard() {
      if (entered && trx != nullptr) {
        TrxInInnoDB::end_stmt(trx);
      }
    }
  } depth_guard{stmt_trx, false};
  TrxInInnoDB::begin_stmt(stmt_trx);
  depth_guard.entered = true;

  auto map_handler_err = [](int rc) -> dberr_t {
    switch (rc) {
      case 0:
        return DB_SUCCESS;
      case HA_ERR_LOCK_WAIT_TIMEOUT:
        return DB_LOCK_WAIT_TIMEOUT;
      case HA_ERR_LOCK_DEADLOCK:
        return DB_DEADLOCK;
      case HA_ERR_KEY_NOT_FOUND:
      case HA_ERR_END_OF_FILE:
        return DB_RECORD_NOT_FOUND;
      default:
        return DB_ERROR;
    }
  };

  if (mysql_table->s != nullptr && mysql_table->s->rec_buff_length > 0) {
    std::memset(mysql_table->record[0], 0, mysql_table->s->rec_buff_length);
  }

  /* Acquire engine-level table lock so InnoDB sees LOCK_IX. */
  bool locked = false;
  auto unlock_if_needed = [&](handler *handler_ptr) {
    if (locked && handler_ptr != nullptr) {
      handler_ptr->ha_external_lock(thd, F_UNLCK);
      locked = false;
    }
  };
  int lock_rc = h->ha_external_lock(thd, F_WRLCK);
  dberr_t lock_err = map_handler_err(lock_rc);
  if (lock_err != DB_SUCCESS) {
    unlock_if_needed(h);
    return lock_err;
  }
  locked = true;

  struct LockGuard {
    handler *h;
    THD *thd;
    bool &locked_ref;
    ~LockGuard() {
      if (locked_ref && h != nullptr) {
        h->ha_external_lock(thd, F_UNLCK);
        locked_ref = false;
      }
    }
  } lock_guard{h, thd, locked};

  if (mysql_table->s->primary_key == MAX_KEY) {
    ib::warn() << "vec_aux_handler_delete: no primary key on aux table";
    return DB_ERROR;
  }

  KEY* pk_info = mysql_table->key_info + mysql_table->s->primary_key;

  /* Make sure the provided PK columns match the aux table PK layout. */
  if (pk_fields != static_cast<ulint>(pk_info->user_defined_key_parts)) {
    ib::warn() << "vec_aux_handler_delete: pk_fields=" << pk_fields
               << " aux_pk_parts=" << pk_info->user_defined_key_parts
               << " dump=" << vec_format_pk_columns_debug(pk_columns, pk_fields);
    return DB_SCHEMA_MISMATCH;
  }

  ib::warn() << "vec_aux_handler_delete: pk columns -> record buffer";
  if (!vec_apply_pk_columns_to_record(pk_columns, clust_index, pk_fields,
                                      pk_info, mysql_table)) {
    ib::warn() << "vec_aux_handler_delete: failed to apply pk_columns to record";
    return DB_SCHEMA_MISMATCH;
  }

  std::vector<uchar> key_buf(static_cast<size_t>(pk_info->key_length));
  key_copy(key_buf.data(), mysql_table->record[0], pk_info, 0);

  int rc = h->ha_index_read_idx_map(
      mysql_table->record[0], mysql_table->s->primary_key, key_buf.data(),
      make_prev_keypart_map(pk_info->user_defined_key_parts),
      HA_READ_KEY_EXACT);
  dberr_t err = map_handler_err(rc);
  if (err != DB_SUCCESS) {
    ib::warn() << "vec_aux_handler_delete: index_read_idx_map failed rc="
               << rc;
    return err;
  }

  Field* faiss_field =
      (pk_fields < static_cast<ulint>(mysql_table->s->fields))
          ? mysql_table->field[pk_fields]
          : nullptr;
  if (faiss_field == nullptr || faiss_field->is_null()) {
    return DB_RECORD_NOT_FOUND;
  }

  const uint64_t vid =
      static_cast<uint64_t>(static_cast<ulonglong>(faiss_field->val_int()));

  rc = h->ha_delete_row(mysql_table->record[0]);
  err = map_handler_err(rc);
  if (err == DB_SUCCESS && out_vid != nullptr) {
    *out_vid = vid;
  }

  return err;
}

/** Extract only the required flags from table->flags2 for VEC aux
tables. Avoid DICT_TF2_AUX so these tables are not misclassified as
FTS auxiliary objects (which triggers FTS-only assertions). Always
tag the table as VECINDEX-aware instead. 
@param[in]      flags2  Table flags2
@return extracted flags2 for VEC aux tables */
static inline uint32_t vec_get_table_flags2_for_aux_tables(uint32_t flags2) {
  /* Extract the file_per_table flag, temporary file flag and encryption flag
  from the main table flags2 */
  return ((flags2 & DICT_TF2_USE_FILE_PER_TABLE) |
          (flags2 & DICT_TF2_ENCRYPTION_FILE_PER_TABLE) |
          (flags2 & DICT_TF2_TEMPORARY));
}

/** Create dict_table_t object for VEC Aux tables.
@param[in]      aux_table_name  VEC Aux table name
@param[in]      table           table object of VEC Index
@param[in]      n_cols          number of columns for VEC Aux table
@return table object for VEC Aux table */
static dict_table_t *vec_create_in_mem_aux_table(const char *aux_table_name,
                                                 const dict_table_t *table,
                                                 ulint n_cols) {
  dict_table_t *new_table = dict_mem_table_create(
      aux_table_name, table->space, n_cols, 0, 0, table->flags,
      vec_get_table_flags2_for_aux_tables(table->flags2));

  if (DICT_TF_HAS_SHARED_SPACE(table->flags)) {
    ut_ad(table->space == fil_space_get_id_by_name(table->tablespace()));
    new_table->tablespace = mem_heap_strdup(new_table->heap, table->tablespace);
  }

  if (DICT_TF_HAS_DATA_DIR(table->flags)) {
    ut_ad(table->data_dir_path != nullptr);
    new_table->data_dir_path =
        mem_heap_strdup(new_table->heap, table->data_dir_path);
  }

  return (new_table);
}



/** Update DD for the single auxiliary table of a VEC index.
@param[in]  index   the vector index instance (already has aux table created)
@return DB_SUCCESS or error code */
static dberr_t vec_create_one_index_dd_tables(const dict_index_t* index)
{
  ut_ad(index != nullptr);
  ut_ad(index->table != nullptr);
  ib::warn() << "VECINDEX: DD register step 1! ";
  // 如果你有 DICT_VECINDEX 标志，做个断言
  ut_ad(index->type & DICT_VECINDEX);

  // Need to get the full name of the aux table, one is _NEXT, the other is _MEM
  // Should receive name in parameter?
  std::string full_name;
  if (index->vec_runtime != nullptr) {
    std::lock_guard<std::shared_mutex> lk(index->vec_runtime->mu);
    if (!index->vec_runtime->pending_aux_name.empty()) {
      full_name = index->vec_runtime->pending_aux_name;
    }
  }
  if (full_name.empty()) {
    full_name = vec_aux_mem_name(vec_aux_full_name(index));
  }

  ib::warn() << "VECINDEX: DD register step 2! full_name=" << full_name;
  // 打开 InnoDB 内部已创建好的物理表（只在内存里用，不登记/修改）
  dict_table_t* aux = dd_table_open_on_name_in_mem(full_name.c_str(), false);
  if (aux == nullptr) {
      ib::warn() << "VECINDEX: DD register failed; cannot open aux table in mem: "
              << full_name;
      return DB_FAIL;
  }
  ib::warn() << "VECINDEX: DD register step 3! ";
  // 把 aux 的定义写进 SQL-DD（实现应仿 dd_create_fts_index_table）
  bool ok = dd_create_vec_index_table(index->table, aux);
  if (!ok) {
      ib::warn() << "VECINDEX: dd_create_vec_index_table() failed for " << full_name;
      dd_table_close(aux, nullptr, nullptr, false);
      return DB_FAIL;
  }
  ib::warn() << "VECINDEX: DD register step 4! ";
  dd_table_close(aux, nullptr, nullptr, false);


  {
    dict_table_t* t = dict_table_open_on_name(full_name.c_str(), false, false,
                                              DICT_ERR_IGNORE_NONE);
    bool dd_ok = (t != nullptr) && (t->dd_space_id != dd::INVALID_OBJECT_ID);

    if (t != nullptr) {
      dict_table_close(t, false, false);
    }

    if (!dd_ok) {
      ib::warn() << "VECINDEX: sanity check failed; cannot open aux table in DD: "
                 << full_name;
      return DB_ERROR;
    }
  }
  ib::warn() << "VECINDEX: DD register Verified! ";

  ib::warn() << "VECINDEX: DD register successed! ";

  return DB_SUCCESS;
}

namespace {
// Convert to key image for mysql engine
inline void vec_cache_append_u32(std::vector<unsigned char> &buf,
                                 uint32_t value) {
  buf.push_back(static_cast<unsigned char>(value & 0xFF));
  buf.push_back(static_cast<unsigned char>((value >> 8) & 0xFF));
  buf.push_back(static_cast<unsigned char>((value >> 16) & 0xFF));
  buf.push_back(static_cast<unsigned char>((value >> 24) & 0xFF));
}

inline bool vec_cache_read_u32(const unsigned char *&p, size_t &remain,
                               uint32_t &out) {
  if (remain < sizeof(uint32_t)) {
    return false;
  }
  out = static_cast<uint32_t>(p[0]) |
        (static_cast<uint32_t>(p[1]) << 8) |
        (static_cast<uint32_t>(p[2]) << 16) |
        (static_cast<uint32_t>(p[3]) << 24);
  p += sizeof(uint32_t);
  remain -= sizeof(uint32_t);
  return true;
}

static bool vec_bind_tuple_from_entry(const unsigned char *data, size_t len,
                                      dict_index_t *clust_index,
                                      dtuple_t *tuple) {
  if (data == nullptr || clust_index == nullptr || tuple == nullptr) {
    return false;
  }

  const unsigned char *p = data;
  size_t remain = len;

  uint32_t num_cols = 0;
  if (!vec_cache_read_u32(p, remain, num_cols)) {
    return false;
  }

  const ulint expected = dict_index_get_n_unique(clust_index);
  const ulint total_fields = clust_index->n_fields;
  if (num_cols != expected) {
    return false;
  }

  dtuple_set_n_fields(tuple, total_fields);
  dtuple_set_n_fields_cmp(tuple, expected);

  for (uint32_t i = 0; i < num_cols; ++i) {
    uint32_t len_field = 0;
    if (!vec_cache_read_u32(p, remain, len_field)) {
      return false;
    }

    dfield_t *df = dtuple_get_nth_field(tuple, i);
    if (len_field == std::numeric_limits<uint32_t>::max()) {
      dfield_set_null(df);
      continue;
    }

    if (remain < len_field) {
      return false;
    }

    dfield_set_data(df, const_cast<unsigned char *>(p), len_field);
    p += len_field;
    remain -= len_field;
  }

  for (ulint i = num_cols; i < total_fields; ++i) {
    dfield_t *df = dtuple_get_nth_field(tuple, i);
    dfield_set_null(df);
  }

  return true;
}

static std::vector<unsigned char> vec_pack_pk_entry(
    const std::vector<vec_pk_column_t> &pk_columns, ulint pk_fields) {
  std::vector<unsigned char> packed;
  const ulint cols = std::min<ulint>(pk_fields, pk_columns.size());
  const uint32_t num_cols = static_cast<uint32_t>(cols);
  size_t total = sizeof(uint32_t);
  for (ulint i = 0; i < cols; ++i) {
    const auto &col = pk_columns[i];
    total += sizeof(uint32_t);
    if (!col.is_null) {
      total += col.data.size();
    }
  }
  packed.reserve(total);

  vec_cache_append_u32(packed, num_cols);
  for (ulint i = 0; i < cols; ++i) {
    const auto &col = pk_columns[i];
    if (col.is_null) {
      vec_cache_append_u32(packed,
                           std::numeric_limits<uint32_t>::max());
      continue;
    }
    const uint32_t len =
        static_cast<uint32_t>(std::min<size_t>(col.data.size(), UINT32_MAX));
    vec_cache_append_u32(packed, len);
    packed.insert(packed.end(), col.data.begin(), col.data.end());
  }

  return packed;
}

constexpr uint32_t VID_PK_MAPPING_MAGIC = 0x4D4B5056;  // "VPKM"
constexpr uint16_t VID_PK_MAPPING_VERSION = 1;

#pragma pack(push, 1)
struct VidPkMappingHeader {
  uint32_t magic;
  uint16_t version;
  uint16_t reserved;
  uint64_t entry_count;
  uint32_t key_length;
  uint8_t  reserved2[4];
};
#pragma pack(pop)

static_assert(sizeof(VidPkMappingHeader) == 24,
              "VidPkMappingHeader size mismatch");

inline bool vec_pk_flush(FILE *fp) {
  if (fp == nullptr) {
    return false;
  }
  if (fflush(fp) != 0) {
    return false;
  }
  const int fd = fileno(fp);
  if (fd < 0) {
    return false;
  }
  return fsync(fd) == 0;
}

}  // namespace

dberr_t vec_insert_aux_cache(vid_pk_mapping_t *cache,
                             dict_index_t *clust_index, uint64_t faiss_id,
                             const std::vector<vec_pk_column_t> &pk_columns) {
  if (cache == nullptr || clust_index == nullptr) {
    return DB_ERROR;
  }

  if (faiss_id >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    ib::error() << "VECINDEX: faiss_id too large for aux cache: " << faiss_id;
    return DB_ERROR;
  }

  const ulint pk_fields = dict_index_get_n_unique(clust_index);
  if (pk_fields == 0 || pk_columns.size() < pk_fields) {
    return DB_ERROR;
  }

  std::vector<unsigned char> packed =
      vec_pack_pk_entry(pk_columns, pk_fields);
  if (packed.empty()) {
    packed.resize(sizeof(uint32_t));
  }

  if (cache->key_length == 0) {
    cache->key_length = packed.size();
  }

  const size_t target = static_cast<size_t>(faiss_id);
  if (target >= cache->pk_values.size()) {
    cache->pk_values.resize(target + 1);
  }
  cache->pk_values[target] = std::move(packed);
  cache->ready = true;

  return DB_SUCCESS;
}

bool vec_aux_cache_bind_tuple(const vid_pk_mapping_t *cache,
                              uint64_t faiss_id, dict_index_t *clust_index,
                              dtuple_t *tuple) {
  if (cache == nullptr || clust_index == nullptr || tuple == nullptr ||
      !cache->ready) {
    return false;
  }

  if (faiss_id >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return false;
  }

  size_t idx = static_cast<size_t>(faiss_id);
  if (idx >= cache->pk_values.size()) {
    return false;
  }

  const std::vector<unsigned char> &entry = cache->pk_values[idx];
  return vec_bind_tuple_from_entry(entry.data(), entry.size(), clust_index,
                                   tuple);
}

bool vec_aux_cache_bind_tuple_copy(const vid_pk_mapping_t *cache,
                                   uint64_t faiss_id,
                                   dict_index_t *clust_index, dtuple_t *tuple,
                                   std::vector<unsigned char> *entry_copy) {
  if (entry_copy == nullptr || cache == nullptr || clust_index == nullptr ||
      tuple == nullptr || !cache->ready) {
    return false;
  }

  if (faiss_id >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return false;
  }

  size_t idx = static_cast<size_t>(faiss_id);
  if (idx >= cache->pk_values.size()) {
    return false;
  }

  const std::vector<unsigned char> &entry = cache->pk_values[idx];
  entry_copy->assign(entry.begin(), entry.end());
  return vec_bind_tuple_from_entry(entry_copy->data(), entry_copy->size(),
                                   clust_index, tuple);
}


std::string vec_vid_pk_mapping_path(const std::string& index_path) {
  if (index_path.empty()) {
    return {};
  }
  std::string path = index_path;
  path.append(".pkmap");
  return path;
}

bool vec_vid_pk_mapping_save(const vid_pk_mapping_t& mapping,
                             const std::string& path) {
  if (!mapping.ready || path.empty()) {
    return false;
  }

  if (mapping.key_length > UINT32_MAX) {
    ib::warn() << "VECINDEX: pk mapping key length too large to persist: "
               << mapping.key_length;
    return false;
  }

  for (const auto& entry : mapping.pk_values) {
    if (entry.size() > UINT32_MAX) {
      ib::warn() << "VECINDEX: pk mapping entry too large to persist ("
                 << entry.size() << " bytes)";
      return false;
    }
  }

  FILE* raw = std::fopen(path.c_str(), "wb");
  if (raw == nullptr) {
    ib::warn() << "VECINDEX: failed to open pk mapping file '" << path << "'";
    return false;
  }
  std::unique_ptr<FILE, decltype(&std::fclose)> fp(raw, &std::fclose);

  VidPkMappingHeader header{};
  header.magic = VID_PK_MAPPING_MAGIC;
  header.version = VID_PK_MAPPING_VERSION;
  header.entry_count = static_cast<uint64_t>(mapping.pk_values.size());
  header.key_length = static_cast<uint32_t>(mapping.key_length);

  if (std::fwrite(&header, sizeof(header), 1, fp.get()) != 1) {
    return false;
  }

  for (const auto& entry : mapping.pk_values) {
    const uint32_t len = static_cast<uint32_t>(entry.size());
    if (std::fwrite(&len, sizeof(len), 1, fp.get()) != 1) {
      return false;
    }
    if (len > 0 &&
        std::fwrite(entry.data(), 1, len, fp.get()) != len) {
      return false;
    }
  }

  if (!vec_pk_flush(fp.get())) {
    ib::warn() << "VECINDEX: failed to flush pk mapping file '" << path << "'";
    return false;
  }

  return true;
}

bool vec_vid_pk_mapping_load(const std::string& path,
                             vid_pk_mapping_t* mapping) {
  if (mapping == nullptr || path.empty()) {
    return false;
  }

  mapping->clear();

  FILE* raw = std::fopen(path.c_str(), "rb");
  if (raw == nullptr) {
    return false;
  }
  std::unique_ptr<FILE, decltype(&std::fclose)> fp(raw, &std::fclose);

  VidPkMappingHeader header{};
  if (std::fread(&header, sizeof(header), 1, fp.get()) != 1) {
    return false;
  }

  if (header.magic != VID_PK_MAPPING_MAGIC ||
      header.version != VID_PK_MAPPING_VERSION) {
    ib::warn() << "VECINDEX: pk mapping header mismatch for '" << path << "'";
    return false;
  }

  off_t file_size = 0;
  const int fd = fileno(fp.get());
  if (fd >= 0) {
    struct stat st {};
    if (fstat(fd, &st) == 0) {
      file_size = st.st_size;
    }
  }

  const uint64_t entry_count = header.entry_count;
  if (entry_count >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    ib::warn() << "VECINDEX: pk mapping entry_count too large in '" << path
               << "'";
    return false;
  }

  const uint64_t header_bytes = sizeof(VidPkMappingHeader);
  if (entry_count >
      (std::numeric_limits<uint64_t>::max() - header_bytes) /
          sizeof(uint32_t)) {
    ib::warn() << "VECINDEX: pk mapping entry_count overflow in '" << path
               << "'";
    return false;
  }

  const uint64_t min_size = header_bytes + entry_count * sizeof(uint32_t);
  if (file_size > 0 && static_cast<uint64_t>(file_size) < min_size) {
    ib::warn() << "VECINDEX: pk mapping file truncated '" << path << "'";
    return false;
  }

  mapping->key_length = static_cast<size_t>(header.key_length);
  mapping->pk_values.resize(static_cast<size_t>(entry_count));

  for (size_t i = 0; i < mapping->pk_values.size(); ++i) {
    uint32_t len = 0;
    if (std::fread(&len, sizeof(len), 1, fp.get()) != 1) {
      mapping->clear();
      return false;
    }
    if (len == 0) {
      continue;
    }
    mapping->pk_values[i].resize(len);
    if (std::fread(mapping->pk_values[i].data(), 1, len, fp.get()) != len) {
      mapping->clear();
      return false;
    }
  }

  mapping->ready = true;
  return true;
}

static std::string vec_aux_next_name(const std::string& prefix) {
  std::string name = prefix;
  name.append("_NEXT");
  return name;
}

static void vec_append_unique(std::vector<std::string>* out,
                              const std::string& value) {
  if (out == nullptr || value.empty()) {
    return;
  }
  if (std::find(out->begin(), out->end(), value) != out->end()) {
    return;
  }
  out->push_back(value);
}

bool vec_collect_drop_resources(dict_table_t* table,
                                vec_aux_drop_resources_t* out) {
  if (table == nullptr || out == nullptr) {
    return false;
  }

  out->clear();

  if (!dict_table_has_vec_index(table)) {
    return true;
  }

  for (dict_index_t* index = table->first_index(); index != nullptr;
       index = index->next()) {
    if (!dict_index_is_vector(index)) {
      continue;
    }

    vec_aux_drop_index_info info{};
    info.table_id = static_cast<uint64_t>(table->id);
    info.index_id = static_cast<uint64_t>(index->id);
    if (index->name != nullptr) {
      info.index_name = index->name;
    }

    info.prefix = vec_aux_prefix(index);
    if (!info.prefix.empty()) {
      info.mem_name = vec_aux_mem_name(info.prefix);
      info.next_name = vec_aux_next_name(info.prefix);
    }else{
      ib::warn() << "VECINDEX: vec_collect_drop_resources: empty aux prefix for index "
                 << (index->name ? index->name : "(null)") << " on table "
                 << (table->name.m_name ? table->name.m_name : "(null)");
      return false;
    }

    std::string meta_path;
    if (vec_meta_path_for_index(index, &meta_path)) {
      info.meta_path = meta_path;
    }

    if (!info.meta_path.empty()) {
      VecMetaHeader header{};
      std::vector<VecSegmentEntry> entries;
      if (vec_meta_read_all(info.meta_path, &header, &entries)) {
        const std::string base_dir = vec_meta_dirname(info.meta_path);
        for (const auto& entry : entries) {
          if (!info.prefix.empty() && entry.seg_id > 0) {
            std::string seg_name = vec_aux_segment_name(
                info.prefix, static_cast<uint32_t>(entry.seg_id));
            vec_append_unique(&info.seg_names, seg_name);
          }
          if (entry.file_name[0] != '\0') {
            std::string seg_path = vec_meta_join(base_dir, entry.file_name);
            vec_append_unique(&info.segment_files, seg_path);
            std::string pkmap_path = vec_vid_pk_mapping_path(seg_path);
            vec_append_unique(&info.pkmap_files, pkmap_path);
          }
        }
      }
    }else{
      ib::warn() << "VECINDEX: vec_collect_drop_resources: failed to read meta for index "
                 << (index->name ? index->name : "(null)") << " on table "
                 << (table->name.m_name ? table->name.m_name : "(null)");
    }

    if ((info.seg_names.empty() || info.segment_files.empty()) &&
        index->vec_runtime != nullptr) {
      vec_index_ctx_t* ctx = index->vec_runtime;
      std::lock_guard<std::shared_mutex> lk(ctx->mu);
      for (const auto& seg : ctx->segments) {
        if (!seg.immutable || seg.vecindex_id == 0) {
          continue;
        }
        std::string seg_name = seg.aux_table_name;
        if (seg_name.empty() && !info.prefix.empty()) {
          seg_name = vec_aux_segment_name(info.prefix, seg.vecindex_id);
        }
        vec_append_unique(&info.seg_names, seg_name);
        if (!seg.index_file_name.empty()) {
          vec_append_unique(&info.segment_files, seg.index_file_name);
          std::string pkmap_path =
              vec_vid_pk_mapping_path(seg.index_file_name);
          vec_append_unique(&info.pkmap_files, pkmap_path);
        }
      }
    }

    if (info.seg_names.empty() && !info.prefix.empty()) {
      const uint32_t max_id = vec_aux_scan_max_segment(info.prefix);
      for (uint32_t seg_id = 1; seg_id <= max_id; ++seg_id) {
        std::string seg_name = vec_aux_segment_name(info.prefix, seg_id);
        if (vec_aux_table_exists(seg_name)) {
          vec_append_unique(&info.seg_names, seg_name);
        }
      }
    }

    out->indexes.push_back(std::move(info));
  }

  return true;
}

dberr_t vec_lock_all_aux_tables(THD* thd,
                                const vec_aux_drop_resources_t* resources) {
  if (thd == nullptr || resources == nullptr) {
    return DB_ERROR;
  }

  std::unordered_set<std::string> names;
  for (const auto& info : resources->indexes) {
    if (!info.mem_name.empty()) {
      names.insert(info.mem_name);
    }
    if (!info.next_name.empty()) {
      names.insert(info.next_name);
    }
    for (const auto& seg_name : info.seg_names) {
      if (!seg_name.empty()) {
        names.insert(seg_name);
      }
    }
  }

  for (const auto& full_name : names) {
    if (full_name.empty() || !vec_is_aux_table_name(full_name.c_str())) {
      continue;
    }
    std::string db_name;
    std::string table_name;
    dict_name::get_table(full_name.c_str(), db_name, table_name);
    if (db_name.empty() || table_name.empty()) {
      return DB_ERROR;
    }

    MDL_ticket* mdl_ticket = nullptr;
    if (dd::acquire_exclusive_table_mdl(thd, db_name.c_str(),
                                        table_name.c_str(), false,
                                        &mdl_ticket)) {
      return DB_ERROR;
    }
  }

  return DB_SUCCESS;
}

static void vec_close_aux_dict_tables(vec_index_ctx_t* ctx) {
  if (ctx == nullptr) {
    return;
  }

  auto close_table = [](dict_table_t*& table) {
    if (table != nullptr) {
      const char* name =
          table->name.m_name != nullptr ? table->name.m_name : "(null)";
      // ib::warn() << "VECREF: before close aux dict table '" << name
      //            << "' ref=" << table->get_ref_count();
#ifdef UNIV_DEBUG
      const bool dict_locked = dict_sys_mutex_own();
#else
      const bool dict_locked = false;
#endif
      dd_table_close(table, nullptr, nullptr, dict_locked);

      // ib::warn() << "VECREF: after close aux dict table '" << name
      //            << "' ref=" << table->get_ref_count();
      table = nullptr;
    }
  };

  std::lock_guard<std::shared_mutex> lk(ctx->mu);
  for (auto& seg : ctx->segments) {
    close_table(seg.aux_dict_table);
  }
  close_table(ctx->staging_segment.aux_dict_table);
  close_table(ctx->pending_aux_dict);
}

static void vec_close_aux_for_table(dict_table_t* table) {
  ib::warn() << "VECINDEX: vec_close_aux_for_table called for table";
             
  if (table == nullptr) {
    ib::warn() << "VECINDEX: vec_close_aux_for_table called with null table";
    return;
  }

  for (dict_index_t* index = table->first_index(); index != nullptr;
       index = index->next()) {
    if (!dict_index_is_vector(index) || index->vec_runtime == nullptr) {
      continue;
    }
    ib::warn() << "VECINDEX: vec_close_aux_for_table closing aux dict tables";
    vec_close_aux_dict_tables(index->vec_runtime);
  }
}

static void vec_aux_push_name(aux_name_vec_t* aux_vec,
                              const std::string& name) {
  if (aux_vec == nullptr || name.empty()) {
    return;
  }
  aux_vec->aux_name.push_back(mem_strdup(name.c_str()));
}

static void vec_aux_drop_files_for_index(const vec_aux_drop_index_info& info) {
  auto drop_one = [](const std::string& path) {
    if (path.empty()) {
      return;
    }
    bool existed = false;
    if (!os_file_delete_if_exists(innodb_data_file_key, path.c_str(), &existed) &&
        existed) {
      ib::warn() << "VECINDEX: failed to delete file '" << path << "'";
    }
  };

  for (const auto& seg_path : info.segment_files) {
    drop_one(seg_path);
  }
  for (const auto& pkmap_path : info.pkmap_files) {
    drop_one(pkmap_path);
  }
  if (!info.meta_path.empty()) {
    drop_one(info.meta_path);
  }
}

dberr_t vec_drop_ancillary_tables(trx_t* trx, dict_table_t* table,
                                  aux_name_vec_t* aux_vec,
                                  const vec_aux_drop_resources_t* resources) {
  const char* table_name = "(null)";
  if (table != nullptr && table->name.m_name != nullptr) {
    table_name = table->name.m_name;
  }
  ib::warn() << "VECINDEX: vec_drop_ancillary_tables enter table '"
             << table_name << "'";
  if (trx == nullptr || table == nullptr) {
    ib::warn() << "VECINDEX: vec_drop_ancillary_tables called with null trx/table";
    return DB_ERROR;
  }
  if (aux_vec != nullptr) {
    vec_free_aux_names(aux_vec);
  }
  const bool has_vec = dict_table_has_vec_index(table);
  const bool is_aux = vec_dict_table_is_aux(table);
  if (!has_vec || is_aux) {
    ib::warn() << "VECINDEX: vec_drop_ancillary_tables skipping table '"
               << (table->name.m_name ? table->name.m_name : "(null)")
               << "' has_vec=" << has_vec << " is_aux=" << is_aux;
    return DB_SUCCESS;
  }

  vec_close_aux_for_table(table);

  vec_aux_drop_resources_t local;
  const vec_aux_drop_resources_t* src = resources;
  if (src == nullptr) {
    ib::warn() << "VECINDEX: vec_drop_ancillary_tables collecting resources for table '"
               << (table->name.m_name ? table->name.m_name : "(null)") << "'";
    if (!vec_collect_drop_resources(table, &local)) {
      ib::warn() << "VECINDEX: vec_drop_ancillary_tables failed to collect resources for table '"
                 << (table->name.m_name ? table->name.m_name : "(null)") << "'";
      return DB_ERROR;
    }
    src = &local;
  }

  ib::warn() << "VECINDEX: vec_drop_ancillary_tables begin table '"
             << (table->name.m_name ? table->name.m_name : "(null)")
             << "' indexes=" << src->indexes.size();
  for (const auto& info : src->indexes) {
    ib::warn() << "VECINDEX: drop plan index_id=" << info.index_id
               << " mem=" << info.mem_name << " next=" << info.next_name
               << " segs=" << info.seg_names.size()
               << " files=" << info.segment_files.size()
               << " pkmap=" << info.pkmap_files.size()
               << " meta=" << info.meta_path;
  }

  ib::warn() << "VECINDEX: vec_drop_ancillary_tables mid: dropping aux tables for table '"
             << table_name << "'";
  for (const auto& info : src->indexes) {
    if (!info.mem_name.empty()) {
      dberr_t err = row_drop_table_for_mysql(info.mem_name.c_str(), trx, false,
                                             nullptr);
      if (err != DB_SUCCESS) {
        ib::warn() << "VECINDEX: failed to drop aux table '"
                   << info.mem_name << "' err=" << err;
        return err;
      }
      vec_aux_push_name(aux_vec, info.mem_name);
    }
    if (!info.next_name.empty()) {
      dberr_t err = row_drop_table_for_mysql(info.next_name.c_str(), trx, false,
                                             nullptr);
      if (err != DB_SUCCESS) {
        ib::warn() << "VECINDEX: failed to drop aux table '"
                   << info.next_name << "' err=" << err;
        return err;
      }
      vec_aux_push_name(aux_vec, info.next_name);
    }
    for (const auto& seg_name : info.seg_names) {
      if (seg_name.empty()) {
        continue;
      }
      dberr_t err = row_drop_table_for_mysql(seg_name.c_str(), trx, false,
                                             nullptr);
      if (err != DB_SUCCESS) {
        ib::warn() << "VECINDEX: failed to drop aux table '"
                   << seg_name << "' err=" << err;
        return err;
      }
      vec_aux_push_name(aux_vec, seg_name);
    }
  }

  ib::warn() << "VECINDEX: vec_drop_ancillary_tables completed for table '"
             << (table->name.m_name ? table->name.m_name : "(null)") << "'";
  return DB_SUCCESS;
}

bool vec_drop_dd_tables(const aux_name_vec_t* aux_vec, bool file_per_table) {
  const size_t aux_count =
      (aux_vec != nullptr) ? aux_vec->aux_name.size() : 0;
  ib::warn() << "VECINDEX: vec_drop_dd_tables enter count=" << aux_count
             << " file_per_table=" << file_per_table;
  if (aux_vec == nullptr || aux_vec->aux_name.empty()) {
    return true;
  }

  ib::warn() << "VECINDEX: vec_drop_dd_tables begin count="
             << aux_vec->aux_name.size()
             << " file_per_table=" << file_per_table;
  ib::warn() << "VECINDEX: vec_drop_dd_tables mid: dropping DD entries";
  bool ok = true;
  for (const auto& name : aux_vec->aux_name) {
    if (!dd_drop_vec_table(name, file_per_table)) {
      ib::warn() << "VECINDEX: dd_drop_vec_table failed for '" << name << "'";
      ok = false;
    }
  }
  if (!ok) {
    ib::warn() << "VECINDEX: vec_drop_dd_tables completed with errors";
  } else {
    ib::warn() << "VECINDEX: vec_drop_dd_tables completed";
  }
  return ok;
}

void vec_free_aux_names(aux_name_vec_t* aux_vec) {
  if (aux_vec == nullptr || aux_vec->aux_name.empty()) {
    return;
  }
  while (!aux_vec->aux_name.empty()) {
    char* name = aux_vec->aux_name.back();
    ut::free(name);
    aux_vec->aux_name.pop_back();
  }
}

void vec_drop_index_files(const vec_aux_drop_resources_t* resources) {
  const size_t index_count =
      (resources != nullptr) ? resources->indexes.size() : 0;
  ib::warn() << "VECINDEX: vec_drop_index_files enter indexes=" << index_count;
  if (resources == nullptr || resources->indexes.empty()) {
    return;
  }

  ib::warn() << "VECINDEX: vec_drop_index_files begin indexes="
             << resources->indexes.size();
  ib::warn() << "VECINDEX: vec_drop_index_files mid: deleting files";
  for (const auto& info : resources->indexes) {
    vec_aux_drop_files_for_index(info);
  }
  ib::warn() << "VECINDEX: vec_drop_index_files completed";
}


/** Check if a table has vector index needs to have its auxiliary index
tables' metadata updated in DD
@param[in,out]  table           table to check
@return DB_SUCCESS or error code */
dberr_t vec_create_index_dd_tables(dict_table_t *table) {
  dberr_t error = DB_SUCCESS;

  for (dict_index_t *index = table->first_index();
       index != nullptr && error == DB_SUCCESS; index = index->next()) {
    if ((index->type & DICT_VECINDEX) && index->fill_dd) {
      ib::warn() << "VECINDEX: creating DD for aux table of index "
                 << (index->name ? index->name : "(null)")
                 << " on table " << (table->name.m_name ? table->name.m_name : "(null)");
      error = vec_create_one_index_dd_tables(index);
      index->fill_dd = false;
    }

    ut_ad(!index->fill_dd);
  }

  return (error);
}


/** 按“基表聚簇索引复制列定义”的规则创建一张 VEC 附属表。
    - 显式主键：逐列复制 mtype/prtype/len/列名，作为 PRIMARY KEY 列集
    - 无显式主键（GEN_CLUST_INDEX）：使用 row_id BIGINT UNSIGNED 作为 PRIMARY KEY
    - 额外加一列：faiss_id BIGINT UNSIGNED
    - 表的 flags 继承自基表（行格式/压缩策略一致）
  @return 成功返回新表指针；失败返回 nullptr（并设置 trx->error_state） */
static dict_table_t* vec_create_one_index_table_pk_compatible(
    trx_t*              trx,
    const dict_index_t* vec_index,         // 该向量索引（含指向基表的 table*）
    const char*         base_table_name,   // "db/table"
    table_id_t          /*table_id*/,      // 未直接使用；表名由 aux_name 唯一化
    const std::string&  aux_name)          // "I_VEC_<tid>_<iid>"
{
  ut_ad(trx != nullptr);
  ut_ad(vec_index != nullptr);
  ut_ad(base_table_name != nullptr);

  dict_table_t* base = vec_index->table;
  if (!base) {
    ib::warn() << "VECINDEX: base table is null";
    return nullptr;
  }

  dict_index_t* clust = base->first_index();
  if (!clust || !clust->is_clustered()) {
    ib::warn() << "VECINDEX: no clustered index found on base table";
    return nullptr;
  }

  mem_heap_t* heap = mem_heap_create(2048, UT_LOCATION_HERE);

  // 1) 计算附属表内部全名：db/aux
  ib::warn() << "vec_create_one_index_table_pk_compatible: compute full name";

  const std::string full_name = vec_make_aux_name(base_table_name, aux_name);
  if (full_name.empty()) {
    ib::warn() << "VECINDEX: cannot extract db from base="
               << (base_table_name ? base_table_name : "(null)");
    mem_heap_free(heap);
    return nullptr;
  }

  // 2) 计算主键列数 / 是否使用 row_id
  ib::warn() << "vec_create_one_index_table_pk_compatible: compute PK columns";
  bool use_row_id = false;
  ulint pk_n_fields = 0;
  if (std::strcmp(clust->name, "GEN_CLUST_INDEX") == 0) {
    use_row_id   = true;
    pk_n_fields  = 1;                  // row_id
  } else {
    use_row_id   = false;
    pk_n_fields  = clust->n_uniq;    // 显式主键列数
  }

  ib::warn() << "VECINDEX: creating aux table " << full_name
             << " with " << (use_row_id ? "row_id" : "PK cols")
             << " as PRIMARY KEY, pk_n_fields=" << pk_n_fields
             << " clustered index name=" << (clust->name ? clust->name : "(null)");

  // 3) 准备 dict_mem_table_create() 所需参数
  const ulint     n_cols       = pk_n_fields + 1;  // + faiss_id


  // 4) in-mem 创建表对象
  dict_table_t* new_table = vec_create_in_mem_aux_table(full_name.c_str(), base, n_cols);

  if (!new_table) {
    ib::warn() << "VECINDEX: dict_mem_table_create failed for " << full_name;
    mem_heap_free(heap);
    return nullptr;
  }

  // 5) 加列：先按主键列拷贝，再追加 faiss_id
  if (use_row_id) {
    // row_id BIGINT UNSIGNED NOT NULL
    dict_mem_table_add_col(new_table, heap,
                           "row_id",
                           DATA_INT,
                           DATA_NOT_NULL | DATA_UNSIGNED,
                           8 /* BIGINT */,
                           true);
  } else {
    for (ulint i = 0; i < pk_n_fields; ++i) {
      dict_field_t*    f        = clust->get_field(i);
      const dict_col_t* c       = f ? f->col : nullptr;
      const char*       colname = f ? static_cast<const char*>(f->name) : nullptr;

      if (!c || !colname) {
        ib::warn() << "VECINDEX: invalid PK meta at i=" << i;
        mem_heap_free(heap);
        return nullptr;
      }

      ib::warn() << "VECINDEX: adding PK col " << colname
                 << " mtype=" << static_cast<int>(c->mtype)
                 << " prtype=" << static_cast<int>(c->prtype)
                 << " len=" << c->len;

      dict_mem_table_add_col(new_table, heap,
                             colname,
                             c->mtype,   // DATA_INT / DATA_VARMYSQL / ...
                             c->prtype,  // NOT_NULL / BINARY / charset bits
                             c->len,     // 变长用最大长度
                             true);
    }
  }

  // 追加 faiss_id BIGINT UNSIGNED NOT NULL（使用 UINT64_MAX 作为未赋值哨兵）
  dict_mem_table_add_col(new_table, heap,
                         "faiss_id",
                         DATA_INT,
                         DATA_NOT_NULL | DATA_UNSIGNED,
                         8 /* BIGINT */,
                         true);

  // 6) 物理建表
  dberr_t err = row_create_table_for_mysql(new_table, nullptr, nullptr, trx, nullptr);
  if (err != DB_SUCCESS) {
    ib::warn() << "VECINDEX: row_create_table_for_mysql failed for " << full_name
               << " err=" << (int)err;
    trx->error_state = err;
    mem_heap_free(heap);
    return nullptr;
  }

  // 7) 建 PRIMARY（聚簇索引）
  {
    dict_index_t* pk = dict_mem_index_create(
        full_name.c_str(),
        "PRIMARY",
        new_table->space,
        DICT_UNIQUE | DICT_CLUSTERED,
        pk_n_fields);

    if (use_row_id) {
      pk->add_field("row_id", 0, true);
    } else {
      for (ulint i = 0; i < pk_n_fields; ++i) {
        dict_field_t* f = clust->get_field(i);
        ulint prefix = f->prefix_len;          // 复制原前缀长度
        bool asc     = f->is_ascending;      // 复制升降序（若你版本没有该方法，按字段 flag 取）
        pk->add_field(f->name, prefix, asc);
      }
    }

    trx_dict_op_t saved = trx_get_dict_operation(trx);
    err = row_create_index_for_mysql(pk, trx, nullptr, nullptr);
    trx->dict_operation = saved;

    if (err != DB_SUCCESS) {
      ib::warn() << "VECINDEX: create PRIMARY on " << full_name
                 << " failed, err=" << (int)err;
      trx->error_state = err;
      mem_heap_free(heap);
      return nullptr;
    }
  }

  // 8) u_faiss_id secondary index disabled (faiss_id is non-indexed).
  
  mem_heap_free(heap);
  return new_table;
}

// --- 入口 ---

dberr_t vec_create_index_tables_low(trx_t* trx,
                                    dict_index_t* index,
                                    const char* table_name,
                                    table_id_t table_id)
{
    if (!trx || !index || !table_name) return DB_FAIL;

    // 唯一且可追溯的附属表名：I_VEC_<table_id>_<index_id>
    const std::string prefix = vec_aux_prefix(index);
    const std::string mem_full = vec_aux_mem_name(prefix);
    std::string aux = vec_aux_suffix_from_full(mem_full);
    if (aux.empty()) {
      std::ostringstream oss;
      oss << "I_VEC_" << static_cast<unsigned long long>(table_id)
          << "_"      << static_cast<unsigned long long>(index->id) << "_MEM";
      aux = oss.str();
    }

    dict_table_t* t = vec_create_one_index_table_pk_compatible(
        trx, index, table_name, table_id, aux);

    ib::warn() << "VECINDEX: creating aux table " << mem_full
                << " for base=" << table_name
                    << " (index_id=" << (unsigned long long)index->id << ")";
    if (!t) return DB_FAIL;

    index->fill_dd = true; // 与 FTS 逻辑一致：请求填充 DD

    ib::warn() << "VECINDEX: created aux table " << mem_full
                << " for base=" << table_name
                << " (index_id=" << (unsigned long long)index->id << ")";

    return DB_SUCCESS;
}
