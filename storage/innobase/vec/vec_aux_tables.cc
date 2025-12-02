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
#include "vec_index_runtime.h"
#include "current_thd.h"    // current_thd
#include "sql/sql_table.h"  // mysql_rename_table
#include "sql/mysqld.h"     // innodb_hton
#include "sql/dd/cache/dictionary_client.h"  // Dictionary_client::Auto_releaser
#include "sql/dd/dd_schema.h"  // Schema_MDL_locker
#include "sql/dd/dictionary.h"  // acquire_exclusive_table_mdl
#include "sql/sql_class.h"  // THD, dd_client()
#include "sql/mdl.h"        // MDL_request
#include "sql/sql_lex.h"    // lex_start / lex_end
#include "sql/dd/string_type.h"  // dd::String_type


#include <cstring>
#include <cstdint>
#include <string>
#include <sstream>
#include <vector>
#include <utility>
#include <cstdio>   // std::snprintf
#include <cstddef>  // size_t
#include <limits>
#include <algorithm>
#include <cstdlib>  // strtoull
#include <mutex>

#include "fts0priv.h"         // fts_parse_sql / fts_eval_sql
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
  dict_table_t *t = dict_table_open_on_name(full_name.c_str(), false, false,
                                            DICT_ERR_IGNORE_NONE);
  if (t != nullptr) {
    dict_table_close(t, false, false);
    return true;
  }
  return false;
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
    std::lock_guard<std::mutex> lk(ctx->mu);
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

/** Extract only the required flags from table->flags2 for FTS Aux
tables.
@param[in]      flags2  Table flags2
@return extracted flags2 for FTS aux tables */
static inline uint32_t vec_get_table_flags2_for_aux_tables(uint32_t flags2) {
  /* Extract the file_per_table flag, temporary file flag and encryption flag
  from the main FTS table flags2 */
  return ((flags2 & DICT_TF2_USE_FILE_PER_TABLE) |
          (flags2 & DICT_TF2_ENCRYPTION_FILE_PER_TABLE) |
          (flags2 & DICT_TF2_TEMPORARY) | DICT_TF2_AUX);
}

/** Create dict_table_t object for FTS Aux tables.
@param[in]      aux_table_name  FTS Aux table name
@param[in]      table           table object of FTS Index
@param[in]      n_cols          number of columns for FTS Aux table
@return table object for FTS Aux table */
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
    std::lock_guard<std::mutex> lk(index->vec_runtime->mu);
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

}  // namespace

dberr_t vec_insert_aux_cache(vec_index_aux_cache_t *cache,
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

bool vec_aux_cache_bind_tuple(const vec_index_aux_cache_t *cache,
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

  const size_t idx = static_cast<size_t>(faiss_id);
  if (idx >= cache->pk_values.size()) {
    return false;
  }

  const std::vector<unsigned char> &entry = cache->pk_values[idx];
  const unsigned char *p = entry.data();
  size_t remain = entry.size();

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
    uint32_t len = 0;
    if (!vec_cache_read_u32(p, remain, len)) {
      return false;
    }

    dfield_t *df = dtuple_get_nth_field(tuple, i);
    if (len == std::numeric_limits<uint32_t>::max()) {
      dfield_set_null(df);
      continue;
    }

    if (remain < len) {
      return false;
    }

    dfield_set_data(df, const_cast<unsigned char *>(p), len);
    p += len;
    remain -= len;
  }

  for (ulint i = num_cols; i < total_fields; ++i) {
    dfield_t *df = dtuple_get_nth_field(tuple, i);
    dfield_set_null(df);
  }

  return true;
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
    - 额外加一列：faiss_id BIGINT UNSIGNED UNIQUE
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

  // 8) 建二级索引：u_faiss_id(faiss_id)
  {
    dict_index_t* uk = dict_mem_index_create(
        full_name.c_str(),
        "u_faiss_id",
        new_table->space,
        0,
        1);
    uk->add_field("faiss_id", 0, true);

    trx_dict_op_t saved = trx_get_dict_operation(trx);
    err = row_create_index_for_mysql(uk, trx, nullptr, nullptr);
    trx->dict_operation = saved;

    if (err != DB_SUCCESS) {
      ib::warn() << "VECINDEX: create unique index u_faiss_id on " << full_name
                 << " failed, err=" << (int)err;
      trx->error_state = err;
      mem_heap_free(heap);
      return nullptr;
    }
  }
  
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
