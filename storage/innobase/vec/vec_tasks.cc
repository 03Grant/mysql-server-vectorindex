// vec_tasks.cc
#include "vec_tasks.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#include <string>

#include "trx0trx.h"
#include "ut0ut.h"
#include "my_sys.h"
#include "my_thread.h"
#include "mysqld.h"
#include "current_thd.h"
#include "ha_innodb.h"
#include "my_bitmap.h"
#include "sql/sql_table.h"
#include "sql/sql_base.h"
#include "sql/sql_class.h"
#include "sql/sql_lex.h"
#include "sql/mdl.h"
#include "sql/sql_thd_internal_api.h"
#include "sql/transaction.h"
#include "dict0dd.h"
#include "dd/cache/dictionary_client.h"
#include "trx0roll.h"
#include "ut0dbg.h"
#include "vec_aux_tables.h"
#include "vec_faiss_factory.h"
#include "vec_hnswlib_factory.h"
#include "vec_index.h"
#include "vec_index_runtime.h"
#include "vec_meta.h"
#include "vec_params.h"
#include "row0mysql.h"
#include "data0type.h"
#include "vec_txn_buf.h"

namespace {

static bool vec_should_skip_task(const dict_index_t* index) {
  if (index == nullptr || index->table == nullptr) {
    return false;
  }
  if (index->table->to_be_dropped) {
    return true;
  }
  return vec_dict_table_is_aux(index->table);
}

static dict_index_t* vec_find_index_by_id(dict_table_t* table,
                                          space_index_t index_id) {
  if (table == nullptr) {
    return nullptr;
  }
  for (dict_index_t* index = table->first_index(); index != nullptr;
       index = index->next()) {
    if (index->id == index_id) {
      return index;
    }
  }
  return nullptr;
}

bool vec_extract_pk_columns(
    TABLE *mysql_table, dict_index_t *clust_index, ulint pk_fields,
    const std::function<Field *(size_t)> &field_resolver,
    std::vector<vec_pk_column_t> &out) {
  if (mysql_table == nullptr || clust_index == nullptr ||
      mysql_table->s == nullptr) {
    return false;
  }

  out.clear();
  out.resize(pk_fields);

  for (ulint i = 0; i < pk_fields; ++i) {
    Field *f = field_resolver(static_cast<size_t>(i));
    if (f == nullptr) {
      return false;
    }

    vec_pk_column_t col{};
    col.is_null = f->is_null();

    dict_field_t *df = clust_index->get_field(static_cast<ulint>(i));
    if (df != nullptr && df->col != nullptr) {
      col.mtype = df->col->mtype;
      col.prtype = df->col->prtype;
    }

    if (!col.is_null) {
      const uchar *ptr = f->data_ptr();
      const uint len = f->data_length();
      if (ptr == nullptr && len != 0) {
        return false;
      }

      dtype_t dtype;
      dtype_set(&dtype, col.mtype, col.prtype, df->col->len);
      dfield_t dfield;
      dfield_set_type(&dfield, &dtype);
      std::vector<byte> tmp(df->col->len + 16);
      byte *end = row_mysql_store_col_in_innobase_format(
          &dfield, tmp.data(), true, ptr, df->col->len,
          dict_table_is_comp(clust_index->table));
      const ulint stored_len =
          static_cast<ulint>(end - static_cast<byte *>(tmp.data()));
      col.data.assign(tmp.data(), tmp.data() + stored_len);
    }

    out[i] = std::move(col);
  }

  return true;
}

bool vec_extract_vector_field(TABLE *mysql_table, size_t vec_col_no,
                              size_t dim, std::vector<float> &out) {
  if (mysql_table == nullptr || mysql_table->field == nullptr ||
      vec_col_no >= mysql_table->s->fields || dim == 0) {
    return false;
  }

  Field *vf = mysql_table->field[vec_col_no];
  if (vf == nullptr || vf->is_null()) {
    return false;
  }

  String tmp;
  String *data = vf->val_str(&tmp);
  if (data == nullptr) {
    return false;
  }
  const char *raw = data->ptr();
  const size_t len = data->length();
  const size_t expected = dim * sizeof(float);
  if (raw == nullptr || len != expected) {
    return false;
  }

  out.resize(dim);
  for (size_t i = 0; i < dim; ++i) {
    float value;
    std::memcpy(&value, raw + i * sizeof(float), sizeof(float));
    if (!std::isfinite(value)) {
      return false;
    }
    out[i] = value;
  }

  return true;
}

std::unique_ptr<IVectorIndex> make_mutable_index(const vec_params_t &params) {
  vec_params_t mem_params = params;
  mem_params.type_tag = VEC_T_FLAT;
  mem_params.size = 0;
  ib::warn() << "VECINDEX: function::make_mutable_index() Before mutable index with backend ";
  switch (mem_params.backend) {
    case BackendType::Faiss:
      return vec_make_faiss_index(mem_params);
    case BackendType::Hnswlib:
      return vec_make_hnswlib_index(mem_params);
    default:
      return nullptr;
  }

  ib::warn() << "VECINDEX: function::make_mutable_index() After mutable index with backend ";
}

bool vec_dump_segment(const vec_index_segment_t *seg, size_t dim,
                      std::vector<float> &out, std::vector<int64_t> &ids) {
  ib::warn() << "VECINDEX: function::vec_dump_segment() Dump segment with dim " << dim;
  if (seg == nullptr || seg->index == nullptr || dim == 0) {
    ib::warn() << "VECINDEX: function::vec_dump_segment() Invalid segment or dimension.";
    return false;
  }
  const size_t n = seg->index->ntotal();
  out.resize(n * dim);
  ids.resize(n);

  for (size_t i = 0; i < n; ++i) {
    if (!seg->index->reconstruct(i, out.data() + i * dim)) {
      return false;
    }
    ids[i] = static_cast<int64_t>(i);
  }
  ib::warn() << "VECINDEX: function::vec_dump_segment() Finished dumping segment.";
  return true;
}

bool vec_resolve_index_path(const std::string &full_name, std::string *out) {
  ib::warn() << "VECINDEX: function::vec_resolve_index_path() Resolving index path for " << full_name;
  if (out == nullptr || full_name.empty()) return false;

  const auto slash = full_name.find('/');
  if (slash == std::string::npos || slash + 1 >= full_name.size()) {
    return false;
  }
  std::string db = full_name.substr(0, slash);
  std::string tbl = full_name.substr(slash + 1);

  char path[FN_REFLEN + 1]{};
  bool truncated = false;
  size_t len =
      build_table_filename(path, sizeof(path) - 1 - reg_ext_length, db.c_str(),
                           tbl.c_str(), "", 0, &truncated);
  if (len == 0 || truncated) {
    return false;
  }
  std::string final_path(path, len);
  final_path.append(".vec");
  *out = std::move(final_path);
  ib::warn() << "VECINDEX: function::vec_resolve_index_path() Resolved index path to " << *out;
  return true;
}

bool vec_bitmap_mark_missing(vecindex_bitmap_t* bitmap,
                             const std::string& aux_table_name,
                             size_t expected_count, THD* thd) {
  ib::warn() << "VECINDEX: function::vec_bitmap_mark_missing() Marking missing entries from aux table " << aux_table_name;
  if (bitmap == nullptr || aux_table_name.empty() || thd == nullptr) {
    return false;
  }

  bitmap->clear();
  if (expected_count == 0) {
    return true;
  }

  const auto slash = aux_table_name.find('/');
  if (slash == std::string::npos || slash == 0 || slash + 1 >= aux_table_name.size()) {
    return false;
  }
  const std::string db = aux_table_name.substr(0, slash);
  const std::string tbl = aux_table_name.substr(slash + 1);

  MDL_request mdl_request;
  MDL_REQUEST_INIT(&mdl_request, MDL_key::TABLE, db.c_str(), tbl.c_str(),
                   MDL_SHARED_READ, MDL_EXPLICIT);
  if (thd->mdl_context.acquire_lock(&mdl_request, 0)) {
    ib::warn() << "VECINDEX: failed to acquire MDL for bitmap load on '"
               << aux_table_name << "'";
    return false;
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
    ib::warn() << "VECINDEX: failed to acquire DD for bitmap load on '"
               << aux_table_name << "'";
    return false;
  }

  char path[FN_REFLEN + 1]{};
  bool truncated = false;
  size_t path_len = build_table_filename(
      path, sizeof(path) - 1 - reg_ext_length, db.c_str(), tbl.c_str(), "", 0,
      &truncated);
  if (path_len == 0 || truncated) {
    ib::warn() << "VECINDEX: failed to build path for bitmap load on '"
               << aux_table_name << "'";
    return false;
  }

  TABLE* mysql_table = open_table_uncached(
      thd, path, db.c_str(), tbl.c_str(), false, true, *dd_table_obj);
  if (mysql_table == nullptr || mysql_table->file == nullptr) {
    ib::warn() << "VECINDEX: open_table_uncached failed for bitmap load on '"
               << aux_table_name << "'";
    return false;
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

  trx_t* trx = thd_to_trx(thd);
  bool allocated_trx = false;
  if (trx == nullptr) {
    trx = innobase_trx_allocate(thd);
    if (trx == nullptr) {
      ib::warn() << "VECINDEX: failed to allocate trx for bitmap load";
      return false;
    }
    thd_to_trx(thd) = trx;
    allocated_trx = true;
  }
  trx_start_if_not_started(trx, true, UT_LOCATION_HERE);

  struct TrxDepthGuard {
    trx_t* trx{nullptr};
    bool entered{false};
    ~TrxDepthGuard() {
      if (entered && trx != nullptr) {
        TrxInInnoDB::end_stmt(trx);
      }
    }
  } depth_guard{trx, false};
  TrxInInnoDB::begin_stmt(trx);
  depth_guard.entered = true;

  handler* h = mysql_table->file;
  if (mysql_table->read_set != nullptr) {
    bitmap_set_all(mysql_table->read_set);
  }

  if (h->ha_rnd_init(true)) {
    ib::warn() << "VECINDEX: ha_rnd_init failed for bitmap load on '"
               << aux_table_name << "'";
    return false;
  }

  struct RndEndGuard {
    handler* h{nullptr};
    ~RndEndGuard() {
      if (h != nullptr) {
        h->ha_rnd_end();
      }
    }
  } rnd_guard{h};

  std::vector<uint8_t> present((expected_count + 7) / 8, 0);

  while (true) {
    int rc = h->ha_rnd_next(mysql_table->record[0]);
    if (rc == HA_ERR_END_OF_FILE) {
      break;
    }
    if (rc != 0) {
      ib::warn() << "VECINDEX: ha_rnd_next failed for bitmap load on '"
                 << aux_table_name << "' rc=" << rc;
      return false;
    }

    if (mysql_table->s->fields == 0) {
      continue;
    }
    Field* faiss_field = mysql_table->field[mysql_table->s->fields - 1];
    if (faiss_field == nullptr || faiss_field->is_null()) {
      continue;
    }
    const ulonglong faiss_id_ull =
        static_cast<ulonglong>(faiss_field->val_int());
    if (faiss_id_ull >= expected_count) {
      continue;
    }
    const size_t idx = static_cast<size_t>(faiss_id_ull);
    present[idx / 8] |= static_cast<uint8_t>(1u << (idx % 8));
  }

  bitmap->clear();
  bitmap->ensure_size(expected_count);
  for (size_t i = 0; i < expected_count; ++i) {
    const size_t byte = i / 8;
    const size_t bit = i % 8;
    if ((present[byte] & static_cast<uint8_t>(1u << bit)) == 0) {
      bitmap->mark(i);
    }
  }

  h->ha_rnd_end();
  rnd_guard.h = nullptr;
  intern_close_table(mysql_table);
  table_guard.t = nullptr;

  if (trx != nullptr &&
      trx->state.load(std::memory_order_relaxed) != TRX_STATE_NOT_STARTED) {
    trx_commit_for_mysql(trx);
  }
  if (allocated_trx) {
    trx_free_for_mysql(trx);
    thd_to_trx(thd) = nullptr;
  }

  return true;
}

static dberr_t vec_backfill_aux_faiss_id(trx_t* trx, dict_index_t* index,
                                         vec_index_segment_t* seg, THD* thd) {
  if (index == nullptr || seg == nullptr || index->table == nullptr ||
      thd == nullptr) {
    return DB_ERROR;
  }

  if (!seg->vid_pk_mapping.ready || seg->vid_pk_mapping.pk_to_vid.empty()) {
    return DB_SUCCESS;
  }

  std::string aux_name = seg->aux_table_name;
  if (aux_name.empty()) {
    vec_index_ctx_t* ctx = index->vec_runtime;
    const std::string prefix =
        (ctx != nullptr && !ctx->index_name_prefix.empty())
            ? ctx->index_name_prefix
            : vec_aux_prefix(index);
    if (prefix.empty()) {
      return DB_ERROR;
    }
    aux_name = seg->immutable ? vec_aux_segment_name(prefix, seg->vecindex_id)
                              : vec_aux_mem_name(prefix);
  }

  const auto slash = aux_name.find('/');
  if (slash == std::string::npos || slash == 0 || slash + 1 >= aux_name.size()) {
    return DB_ERROR;
  }
  const std::string db = aux_name.substr(0, slash);
  const std::string tbl = aux_name.substr(slash + 1);

  MDL_request mdl_request;
  MDL_REQUEST_INIT(&mdl_request, MDL_key::TABLE, db.c_str(), tbl.c_str(),
                   MDL_SHARED_WRITE, MDL_EXPLICIT);
  if (thd->mdl_context.acquire_lock(&mdl_request,
                                    thd->variables.lock_wait_timeout)) {
    ib::warn() << "VECINDEX: failed to acquire MDL for aux backfill on '"
               << aux_name << "'";
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
    ib::warn() << "VECINDEX: failed to acquire DD for aux backfill on '"
               << aux_name << "'";
    return DB_ERROR;
  }

  char path[FN_REFLEN + 1]{};
  bool truncated = false;
  size_t path_len = build_table_filename(
      path, sizeof(path) - 1 - reg_ext_length, db.c_str(), tbl.c_str(), "", 0,
      &truncated);
  if (path_len == 0 || truncated) {
    return DB_ERROR;
  }

  TABLE* mysql_table = open_table_uncached(
      thd, path, db.c_str(), tbl.c_str(), false, true, *dd_table_obj);
  if (mysql_table == nullptr || mysql_table->file == nullptr) {
    return DB_ERROR;
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
      default:
        return DB_ERROR;
    }
  };

  bool locked = false;
  int lock_rc = h->ha_external_lock(thd, F_WRLCK);
  dberr_t lock_err = map_handler_err(lock_rc);
  if (lock_err != DB_SUCCESS) {
    return lock_err;
  }
  locked = true;

  struct LockGuard {
    handler* h;
    THD* thd;
    bool& locked_ref;
    ~LockGuard() {
      if (locked_ref && h != nullptr) {
        h->ha_external_lock(thd, F_UNLCK);
        locked_ref = false;
      }
    }
  } lock_guard{h, thd, locked};

  if (h->ha_rnd_init(true)) {
    ib::warn() << "VECINDEX: ha_rnd_init failed for aux backfill on '"
               << aux_name << "'";
    return DB_ERROR;
  }

  struct RndEndGuard {
    handler* h{nullptr};
    ~RndEndGuard() {
      if (h != nullptr) {
        h->ha_rnd_end();
      }
    }
  } rnd_guard{h};

  dict_index_t* clust_index = index->table->first_index();
  if (clust_index == nullptr) {
    return DB_ERROR;
  }

  const ulint pk_fields = dict_index_get_n_unique(clust_index);
  if (pk_fields == 0 || pk_fields >= mysql_table->s->fields) {
    return DB_ERROR;
  }

  Field* faiss_field = mysql_table->field[pk_fields];
  if (faiss_field == nullptr || mysql_table->record[1] == nullptr) {
    return DB_ERROR;
  }

  size_t updated = 0;
  std::shared_lock<std::shared_mutex> seg_lock(*seg->rw_lock);
  while (true) {
    int rc = h->ha_rnd_next(mysql_table->record[0]);
    if (rc == HA_ERR_END_OF_FILE) {
      break;
    }
    if (rc != 0) {
      ib::warn() << "VECINDEX: ha_rnd_next failed for aux backfill on '"
                 << aux_name << "' rc=" << rc;
      return DB_ERROR;
    }

    std::vector<vec_pk_column_t> pk_cols;
    bool pk_ok = vec_extract_pk_columns(
        mysql_table, clust_index, pk_fields,
        [mysql_table](size_t idx) {
          return (idx < static_cast<size_t>(mysql_table->s->fields))
                     ? mysql_table->field[idx]
                     : nullptr;
        },
        pk_cols);
    if (!pk_ok) {
      return DB_ERROR;
    }

    std::string key = vec_pack_pk_key(pk_cols, pk_fields);
    if (key.empty()) {
      continue;
    }
    auto it = seg->vid_pk_mapping.pk_to_vid.find(key);
    if (it == seg->vid_pk_mapping.pk_to_vid.end()) {
      continue;
    }
    const uint64_t new_vid = it->second;
    if (!faiss_field->is_null()) {
      const ulonglong current =
          static_cast<ulonglong>(faiss_field->val_int());
      if (current == new_vid) {
        continue;
      }
    }

    if (mysql_table->s != nullptr && mysql_table->s->rec_buff_length > 0) {
      std::memcpy(mysql_table->record[1], mysql_table->record[0],
                  mysql_table->s->rec_buff_length);
    }

    faiss_field->store(static_cast<longlong>(new_vid), true);
    rc = h->ha_update_row(mysql_table->record[1], mysql_table->record[0]);
    dberr_t err = map_handler_err(rc);
    if (err != DB_SUCCESS) {
      return err;
    }
    ++updated;
  }

  ib::warn() << "VECINDEX: aux backfill updated rows=" << updated
             << " table=" << aux_name;
  return DB_SUCCESS;
}

bool vec_try_rotate_mem_index(trx_t *trx, dict_index_t *index,
                              vec_index_ctx_t *ctx) {
  ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() 1 Trying to rotate mem index.";
  if (trx == nullptr || index == nullptr || ctx == nullptr) {
    ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() Invalid arguments.";
    return false;
  }

  ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() 2 Trying to rotate mem index.";
  std::unique_lock<std::shared_mutex> lk(ctx->mu);
  vec_index_segment_t *mutable_seg = ctx->mutable_segment();
  if (mutable_seg == nullptr || mutable_seg->index == nullptr) {
    ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() No mutable segment found.";
    return false;
  }

  ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() 3 Trying to rotate mem index.";
  const std::string prefix =
      ctx->index_name_prefix.empty() ? vec_aux_prefix(index)
                                     : ctx->index_name_prefix;
  if (prefix.empty()) {
    ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() Empty prefix.";
    return false;
  }

  ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() 4 Trying to rotate mem index.";
  const uint32_t seg_id =
      ctx->max_vecindex_id == 0 ? 1 : ctx->max_vecindex_id + 1;
  const std::string seg_name = vec_aux_segment_name(prefix, seg_id);
  const std::string mem_name = mutable_seg->aux_table_name.empty()
                                   ? vec_aux_mem_name(prefix)
                                   : mutable_seg->aux_table_name;
  ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() 5 Renaming mem table " << mem_name << " to seg table " << seg_name;
  dberr_t err = vec_aux_rename_table(trx, mem_name, seg_name);
  if (err != DB_SUCCESS) {
    ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() Failed to rename table from " << mem_name << " to " << seg_name;
    return false;
  }

  ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() 6 Creating/renaming new mem table.";

  const std::string new_mem_name = vec_aux_mem_name(prefix);
  std::string pending_name = ctx->pending_aux_name;
  if (pending_name.empty()) {
    ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() No pending aux table to rename to "
               << new_mem_name;
    return false;
  }
  err = vec_aux_rename_table(trx, pending_name, new_mem_name);
  if (err != DB_SUCCESS) {
    ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() Failed to rename pending table from "
               << pending_name << " to " << new_mem_name;
    return false;
  }
  ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() 8 Creating new mutable index.";
  auto new_index = make_mutable_index(ctx->params);
  if (!new_index) {
    ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() Failed to create mutable index.";
    return false;
  }

  vec_index_segment_t staging = std::move(*mutable_seg);
  staging.immutable = true;
  staging.aux_table_name = seg_name;
  staging.vecindex_id = seg_id;

  vec_index_segment_t fresh{};
  fresh.index = std::move(new_index);
  fresh.aux_table_name = new_mem_name;
  fresh.immutable = false;
  fresh.aux_dict_table =
      dd_table_open_on_name_in_mem(new_mem_name.c_str(), false);
  // if (fresh.aux_dict_table != nullptr) {
  //   ib::warn() << "VECREF: open fresh aux dict table '" << new_mem_name
  //              << "' ref=" << fresh.aux_dict_table->get_ref_count();
  // } else {
  //   ib::warn() << "VECREF: failed to open fresh aux dict table '"
  //              << new_mem_name << "'";
  // }
  fresh.vecindex_bitmap.clear();

  if (!ctx->segments.empty()) {
    ctx->segments.erase(ctx->segments.begin());
  }
  ctx->segments.insert(ctx->segments.begin(), std::move(fresh));
  ctx->segments.push_back(std::move(staging));
  ctx->max_vecindex_id = std::max(ctx->max_vecindex_id, seg_id);
  if (ctx->pending_aux_dict != nullptr) {
    const char* pending_name =
        ctx->pending_aux_dict->name.m_name != nullptr
            ? ctx->pending_aux_dict->name.m_name
            : "(null)";
    // ib::warn() << "VECREF: close pending aux dict table '" << pending_name
    //            << "' ref=" << ctx->pending_aux_dict->get_ref_count();
    dd_table_close(ctx->pending_aux_dict, nullptr, nullptr, false);
    ctx->pending_aux_dict = nullptr;
  }
  ctx->pending_aux_name.clear();
  ctx->needs_aux_refresh.store(true);
  lk.unlock();
  ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() Successfully rotated mem index.";
  return true;
}

class VecBgThdGuard {
 public:
  VecBgThdGuard() = default;
  ~VecBgThdGuard() { cleanup(); }

  bool create() {
    prev_thd_ = current_thd;
    thd_ = create_thd(false, true, true, 0, 0);
    if (thd_ == nullptr) {
      current_thd = prev_thd_;
      return false;
    }
    current_thd = thd_;
    thd_->set_command(COM_DAEMON);
    // DD helpers (Update_dictionary_tables_ctx) require autocommit off.
    thd_->variables.option_bits &= ~OPTION_AUTOCOMMIT;
    thd_->variables.option_bits |= OPTION_NOT_AUTOCOMMIT;
    thd_->variables.transaction_read_only = false;
    thd_->tx_read_only = false;
    thd_->system_thread = SYSTEM_THREAD_BACKGROUND;
    thd_->security_context()->skip_grants();
    // DD helpers invoked below expect a DDL-ish sql_command to be present;
    // set a benign default to avoid debug assertions in transactional DDL paths.
    thd_->lex->sql_command = SQLCOM_CREATE_TABLE;
    return true;
  }

  bool start_trx() {
    if (thd_ == nullptr) {
      return false;
    }
    trx_t *&trx_ref = thd_to_trx(thd_);
    if (trx_ref == nullptr) {
      trx_ref = innobase_trx_allocate(thd_);
      owns_trx_ = true;
    } else {
      owns_trx_ = false;
    }
    trx_ = trx_ref;
    if (trx_ == nullptr) {
      return false;
    }
    trx_start_if_not_started(trx_, true, UT_LOCATION_HERE);
    return true;
  }

  bool commit() {
    if (trx_ == nullptr || thd_ == nullptr) {
      return false;
    }
    if (trx_commit_for_mysql(trx_) != DB_SUCCESS) {
      trans_rollback(thd_);
      return false;
    }
    if (trans_commit_stmt(thd_, false)) {
      trans_rollback(thd_);
      return false;
    }
    trans_commit(thd_);
    return true;
  }

  void rollback() {
    if (trx_ != nullptr) {
      trx_rollback_to_savepoint(trx_, nullptr);
    }
    if (thd_ != nullptr) {
      trans_rollback_stmt(thd_);
      trans_rollback(thd_);
    }
  }

  trx_t *trx() const { return trx_; }
  THD *thd() const { return thd_; }

 private:
  void cleanup() {
    if (thd_ != nullptr) {
      current_thd = thd_;
      destroy_thd(thd_);
      thd_ = nullptr;
      current_thd = prev_thd_;
    }
    trx_ = nullptr;
    owns_trx_ = false;
  }

  THD *thd_{nullptr};
  trx_t *trx_{nullptr};
  THD *prev_thd_{nullptr};
  bool owns_trx_{false};
};

bool vec_bg_build_task(dict_index_t *index, vec_index_ctx_t *ctx,
                       vec_index_segment_t *seg) {
  ib::warn() << "VECINDEX: function::vec_bg_build_task() Building index segment in bg.";
  if (index == nullptr || ctx == nullptr || seg == nullptr) {
    ib::warn() << "VECINDEX: function::vec_bg_build_task() Invalid arguments.";
    return false;
  }
  if (vec_should_skip_task(index)) {
    ib::warn() << "VECINDEX: function::vec_bg_build_task() Skip build for dropping/aux table.";
    return false;
  }
  ctx->build_in_progress = true;
  auto clear_build_flag = [&]() {
    ctx->build_in_progress = false;
  };

  std::vector<float> xb;
  std::vector<int64_t> ids;
  vid_pk_mapping_t new_mapping;
  std::vector<int64_t> old_to_new;
  std::vector<uint8_t> bitmap_snapshot;
  const size_t dim = ctx->params.dim;
  {
    std::shared_lock<std::shared_mutex> seg_lock;
    if (seg->rw_lock) {
      seg_lock = std::shared_lock<std::shared_mutex>(*seg->rw_lock);
    }
    if (dim == 0 || seg->index == nullptr) {
      clear_build_flag();
      ib::warn() << "VECINDEX: function::vec_bg_build_task() Invalid segment data.";
      return false;
    }
    const size_t n = seg->index->ntotal();
    bitmap_snapshot = seg->vecindex_bitmap.bits;
    old_to_new.assign(n, -1);
    auto is_marked = [&](size_t idx) -> bool {
      const size_t byte = idx / 8;
      const size_t bit = idx % 8;
      if (byte >= bitmap_snapshot.size()) {
        return false;
      }
      return (bitmap_snapshot[byte] >> bit) & 0x01;
    };

    size_t alive = 0;
    for (size_t i = 0; i < n; ++i) {
      if (!is_marked(i)) {
        ++alive;
      }
    }

    xb.reserve(alive * dim);
    ids.reserve(alive);
    new_mapping.key_length = seg->vid_pk_mapping.key_length;
    new_mapping.pk_values.reserve(alive);

    const auto &pk_values = seg->vid_pk_mapping.pk_values;
    for (size_t i = 0; i < n; ++i) {
      if (is_marked(i)) {
        continue;
      }
      if (i >= pk_values.size() || pk_values[i].empty()) {
        clear_build_flag();
        ib::warn() << "VECINDEX: missing pk mapping for seg_id="
                   << seg->vecindex_id << " vid=" << i;
        return false;
      }

      const size_t new_id = ids.size();
      xb.resize((new_id + 1) * dim);
      if (!seg->index->reconstruct(i, xb.data() + new_id * dim)) {
        clear_build_flag();
        ib::warn() << "VECINDEX: failed to reconstruct seg_id="
                   << seg->vecindex_id << " vid=" << i;
        return false;
      }
      ids.push_back(static_cast<int64_t>(new_id));
      new_mapping.pk_values.push_back(pk_values[i]);
      if (new_mapping.key_length == 0) {
        new_mapping.key_length = new_mapping.pk_values.back().size();
      }
      const auto &entry = new_mapping.pk_values.back();
      std::string key(reinterpret_cast<const char *>(entry.data()), entry.size());
      new_mapping.pk_to_vid[key] = static_cast<uint64_t>(new_id);
      old_to_new[i] = static_cast<int64_t>(new_id);
    }

    new_mapping.is_mem_diff = false;
    new_mapping.mem_diff.clear();
    new_mapping.ready = true;
  }

  std::string index_path;
  if (!vec_resolve_index_path(seg->aux_table_name, &index_path)) {
    clear_build_flag();
    ib::warn() << "VECINDEX: function::vec_bg_build_task() Failed to resolve index path.";
    return false;
  }

  // Prepare metadata entry (append-only; crash recovery TODO).
  std::string meta_path;
  VecMetaFile meta_file;
  VecSegmentEntry meta_entry{};
  long meta_offset = -1;
  bool meta_ready = false;
  if (vec_meta_path_for_index(index, &meta_path)) {
    VecMetaHeader header = vec_meta_make_header(index, ctx->params);
    if (meta_file.open_or_create(meta_path, header)) {
      vec_meta_fill_entry(&meta_entry,
                          static_cast<uint64_t>(seg->vecindex_id),
                          static_cast<uint64_t>(ids.size()),
                          VecSegmentState::Preparing,
                          vec_meta_basename(index_path));
      meta_ready = meta_file.append(&meta_entry, &meta_offset);
    } else {
      ib::warn() << "VECMETA: unable to open meta file '" << meta_path << "'";
    }
  } else {
    ib::warn() << "VECMETA: failed to derive meta path for index "
               << (index->name ? index->name : "(null)");
  }

  auto target = ctx->params.backend == BackendType::Faiss
                    ? vec_make_faiss_index(ctx->params)
                    : vec_make_hnswlib_index(ctx->params);
  if (!target) {
    clear_build_flag();
    ib::warn() << "VECINDEX: function::vec_bg_build_task() Failed to create target index.";
    return false;
  }

  if (!ids.empty()) {
    target->train(ids.size(), xb.data());
    target->add(ids.size(), xb.data(), ids.data());
  }
  
  target->save(index_path);
  const std::string mapping_path = vec_vid_pk_mapping_path(index_path);
  bool pk_mapping_saved = false;
  if (!mapping_path.empty()) {
    pk_mapping_saved =
        vec_vid_pk_mapping_save(new_mapping, mapping_path);
    if (!pk_mapping_saved) {
      ib::warn() << "VECINDEX: failed to persist PK mapping to '"
                 << mapping_path << "'";
    }
  }

  {
    std::unique_lock<std::shared_mutex> seg_lock;
    if (seg->rw_lock) {
      seg_lock = std::unique_lock<std::shared_mutex>(*seg->rw_lock);
    }
    const std::vector<uint8_t> current_bits = seg->vecindex_bitmap.bits;
    auto is_marked_current = [&](size_t idx) -> bool {
      const size_t byte = idx / 8;
      const size_t bit = idx % 8;
      if (byte >= current_bits.size()) {
        return false;
      }
      return (current_bits[byte] >> bit) & 0x01;
    };
    seg->index_file_name = index_path;
    seg->index = std::move(target);
    seg->vecindex_bitmap.clear();
    seg->vecindex_bitmap.ensure_size(new_mapping.pk_values.size());
    for (size_t i = 0; i < old_to_new.size(); ++i) {
      if (old_to_new[i] < 0) {
        continue;
      }
      if (is_marked_current(i)) {
        seg->vecindex_bitmap.mark(static_cast<size_t>(old_to_new[i]));
      }
    }
    seg->vid_pk_mapping = std::move(new_mapping);
  }

  THD *thd = current_thd;
  if (thd != nullptr) {
    trx_t *trx = thd_to_trx(thd);
    dberr_t backfill_err =
        vec_backfill_aux_faiss_id(trx, index, seg, thd);
    // if (backfill_err != DB_SUCCESS) {
    //   ib::warn() << "VECINDEX: aux backfill failed err=" << backfill_err;
    //   (void)trans_rollback_stmt(thd);
    //   (void)trans_rollback(thd);
    // } else {
    //   bool commit_ok = !trans_commit_stmt(thd, false);
    //   if (commit_ok) {
    //     commit_ok = !trans_commit(thd, false);
    //   }
    //   if (!commit_ok) {
    //     ib::warn() << "VECINDEX: aux backfill commit failed";
    //     (void)trans_rollback_stmt(thd);
    //     (void)trans_rollback(thd);
    //   }
    // }
  } else {
    ib::warn() << "VECINDEX: aux backfill skipped (no THD)";
  }

  if (meta_ready) {
    if (pk_mapping_saved) {
      vec_meta_mark_pk_mapping(&meta_entry, true);
    }
    meta_entry.state = static_cast<uint8_t>(VecSegmentState::Committed);
    if (!meta_file.overwrite(meta_offset, &meta_entry)) {
      ib::warn() << "VECMETA: failed to commit meta entry for seg "
                 << seg->vecindex_id;
    }
  }

  clear_build_flag();
  ib::warn() << "VECINDEX: function::vec_bg_build_task() Finished building index segment.";
  return true;
}

bool vec_rotate_mem_index_bg(table_id_t table_id, space_index_t index_id) {
  ib::warn() << "VECINDEX: function::vec_rotate_mem_index_bg() Rotating mem index in bg.";
  bool rename_committed = false;
  bool bg_built = false;

  struct PendingGuard {
    vec_index_ctx_t* ctx{nullptr};
    ~PendingGuard() {
      if (ctx != nullptr) {
        ctx->is_rotation_pending.store(false);
      }
    }
  } pending_guard{};

  {
    VecBgThdGuard guard;
    if (!guard.create()) {
      ib::warn() << "VECINDEX: function::vec_rotate_mem_index_bg() Failed to create background thread.";
      return false;
    }
    if (!guard.start_trx()) {
      ib::warn() << "VECINDEX: function::vec_rotate_mem_index_bg() Failed to start transaction.";
      return false;
    }

    MDL_ticket* mdl = nullptr;
    dict_table_t* table =
        dd_table_open_on_id(table_id, guard.thd(), &mdl, false, true);
    if (table == nullptr) {
      guard.rollback();
      return false;
    }

    dict_index_t* index = vec_find_index_by_id(table, index_id);
    if (index == nullptr || index->vec_runtime == nullptr) {
      dd_table_close(table, guard.thd(), &mdl, false);
      guard.rollback();
      return false;
    }

    vec_index_ctx_t* ctx = index->vec_runtime;
    pending_guard.ctx = ctx;

    if (vec_should_skip_task(index)) {
      dd_table_close(table, guard.thd(), &mdl, false);
      guard.rollback();
      ib::warn() << "VECINDEX: function::vec_rotate_mem_index_bg() Skip rotation for dropping/aux table.";
      return false;
    }

    bool rotated = vec_try_rotate_mem_index(guard.trx(), index, ctx);
    if (rotated) {
      rename_committed = guard.commit();
      if (!rename_committed) {
        guard.rollback();
      }
    } else {
      guard.rollback();
    }

    dd_table_close(table, guard.thd(), &mdl, false);
  }

  if (rename_committed) {
    VecBgThdGuard build_guard;
    if (!build_guard.create()) {
      ib::warn() << "VECINDEX: function::vec_rotate_mem_index_bg() Failed to create build thread.";
    } else if (!build_guard.start_trx()) {
      ib::warn() << "VECINDEX: function::vec_rotate_mem_index_bg() Failed to start build transaction.";
    } else {
      MDL_ticket* mdl = nullptr;
      dict_table_t* table =
          dd_table_open_on_id(table_id, build_guard.thd(), &mdl, false, true);
      if (table == nullptr) {
        build_guard.rollback();
      } else {
        dict_index_t* index = vec_find_index_by_id(table, index_id);
        if (index == nullptr || index->vec_runtime == nullptr) {
          dd_table_close(table, build_guard.thd(), &mdl, false);
          build_guard.rollback();
        } else if (vec_should_skip_task(index)) {
          dd_table_close(table, build_guard.thd(), &mdl, false);
          build_guard.rollback();
          ib::warn() << "VECINDEX: function::vec_rotate_mem_index_bg() Skip build for dropping/aux table.";
        } else {
          vec_index_ctx_t* ctx = index->vec_runtime;
          vec_index_segment_t* staging_seg = nullptr;
          {
            std::lock_guard<std::shared_mutex> lk(ctx->mu);
            if (!ctx->segments.empty()) {
              staging_seg = &ctx->segments.back();
            }
          }
          if (staging_seg != nullptr && staging_seg->immutable) {
            bg_built = vec_bg_build_task(index, ctx, staging_seg);
          }

          if (bg_built) {
            if (!build_guard.commit()) {
              bg_built = false;
            }
          } else {
            build_guard.rollback();
          }
          dd_table_close(table, build_guard.thd(), &mdl, false);
        }
      }
    }
  }

  const bool ok = rename_committed && bg_built;
  ib::warn() << "VECINDEX: function::vec_rotate_mem_index_bg() Rotation "
             << (ok ? "succeeded." : "failed.");
  return ok;
}

}  // namespace

bool vec_load_aux_cache_for_segment(dict_index_t *vec_index,
                                    vec_index_ctx_t *ctx,
                                    vec_index_segment_t *seg, THD *thd) {
  if (vec_index == nullptr || vec_index->table == nullptr || ctx == nullptr ||
      seg == nullptr || thd == nullptr) {
    return false;
  }

  if (seg->vid_pk_mapping.ready) {
    // Usually this way
    return true;
  }

  std::string aux_name = seg->aux_table_name;
  if (aux_name.empty()) {
    const std::string prefix =
        ctx->index_name_prefix.empty() ? vec_aux_prefix(vec_index)
                                       : ctx->index_name_prefix;
    aux_name = vec_aux_segment_name(prefix, seg->vecindex_id);
    if (aux_name.empty()) {
      return false;
    }
    seg->aux_table_name = aux_name;
  }


  const std::string mapping_path =
      vec_vid_pk_mapping_path(seg->index_file_name);
  if (!mapping_path.empty() &&
      vec_vid_pk_mapping_load(mapping_path, &seg->vid_pk_mapping)) {
    bool bitmap_ok = vec_bitmap_mark_missing(
        &seg->vecindex_bitmap, aux_name, seg->vid_pk_mapping.size(), thd);
    if (!bitmap_ok) {
      ib::warn() << "VECINDEX: failed to refresh deletion bitmap from aux table '"
                 << aux_name << "'";
    }
    ib::warn() << "VECINDEX: loaded PK mapping file '" << mapping_path
               << "' rows=" << seg->vid_pk_mapping.size();
    if (bitmap_ok) {
      seg->vid_pk_mapping.ready = true;
      return true;
    }
  }

  dict_index_t *clust_index = vec_index->table->first_index();
  if (clust_index == nullptr) {
    return false;
  }

  const size_t pk_fields = dict_index_get_n_unique(clust_index);
  if (pk_fields == 0) {
    return false;
  }

  const auto slash = aux_name.find('/');
  if (slash == std::string::npos || slash == 0 || slash + 1 >= aux_name.size()) {
    return false;
  }
  const std::string db = aux_name.substr(0, slash);
  const std::string tbl = aux_name.substr(slash + 1);

  MDL_request mdl_request;
  MDL_REQUEST_INIT(&mdl_request, MDL_key::TABLE, db.c_str(), tbl.c_str(),
                   MDL_SHARED_READ, MDL_EXPLICIT);
  if (thd->mdl_context.acquire_lock(&mdl_request, 0)) {
    ib::warn() << "VECINDEX: failed to acquire MDL for aux table '" << aux_name
               << "'";
    return false;
  }

  struct MdlGuard {
    THD *thd{nullptr};
    MDL_request *req{nullptr};
    ~MdlGuard() {
      if (thd != nullptr && req != nullptr && req->ticket != nullptr) {
        thd->mdl_context.release_lock(req->ticket);
        req->ticket = nullptr;
      }
    }
  } mdl_guard{thd, &mdl_request};

  dd::cache::Dictionary_client *client = dd::get_dd_client(thd);
  dd::cache::Dictionary_client::Auto_releaser releaser(client);

  const dd::Table *dd_table_obj = nullptr;
  if (client->acquire<dd::Table>(db.c_str(), tbl.c_str(), &dd_table_obj) ||
      dd_table_obj == nullptr) {
    ib::warn() << "VECINDEX: failed to acquire DD for aux table '" << aux_name
               << "'";
    return false;
  }

  char path[FN_REFLEN + 1]{};
  bool truncated = false;
  size_t path_len = build_table_filename(
      path, sizeof(path) - 1 - reg_ext_length, db.c_str(), tbl.c_str(), "", 0,
      &truncated);
  if (path_len == 0 || truncated) {
    ib::warn() << "VECINDEX: failed to build path for aux table '" << aux_name
               << "'";
    return false;
  }

  TABLE *mysql_table = open_table_uncached(
      thd, path, db.c_str(), tbl.c_str(), false, true, *dd_table_obj);
  if (mysql_table == nullptr || mysql_table->file == nullptr) {
    ib::warn() << "VECINDEX: open_table_uncached failed for aux table '"
               << aux_name << "'";
    return false;
  }

  struct TableGuard {
    TABLE *t{nullptr};
    ~TableGuard() {
      if (t != nullptr) {
        intern_close_table(t);
        t = nullptr;
      }
    }
  } table_guard{mysql_table};

  /* Ensure we have an active transaction and InnoDB depth for handler scan. */
  trx_t *trx = thd_to_trx(thd);
  bool allocated_trx = false;
  if (trx == nullptr) {
    trx = innobase_trx_allocate(thd);
    if (trx == nullptr) {
      ib::warn() << "VECINDEX: failed to allocate trx for aux cache load";
      return false;
    }
    thd_to_trx(thd) = trx;
    allocated_trx = true;
  }
  trx_start_if_not_started(trx, true, UT_LOCATION_HERE);

  struct TrxDepthGuard {
    trx_t *trx{nullptr};
    bool entered{false};
    ~TrxDepthGuard() {
      if (entered && trx != nullptr) {
        TrxInInnoDB::end_stmt(trx);
      }
    }
  } depth_guard{trx, false};
  TrxInInnoDB::begin_stmt(trx);
  depth_guard.entered = true;

  handler *h = mysql_table->file;
  if (mysql_table->read_set != nullptr) {
    bitmap_set_all(mysql_table->read_set);
  }

  if (pk_fields >= mysql_table->s->fields) {
    ib::warn() << "VECINDEX: aux table layout mismatch for '" << aux_name
               << "' pk_fields=" << pk_fields
               << " total_fields=" << mysql_table->s->fields;
    return false;
  }

  if (h->ha_rnd_init(true)) {
    intern_close_table(mysql_table);
    ib::warn() << "VECINDEX: ha_rnd_init failed for aux table '" << aux_name
               << "'";
    return false;
  }

  struct RndEndGuard {
    handler *h{nullptr};
    ~RndEndGuard() {
      if (h != nullptr) {
        h->ha_rnd_end();
      }
    }
  } rnd_guard{h};

  seg->vid_pk_mapping.clear();

  while (true) {
    int rc = h->ha_rnd_next(mysql_table->record[0]);
    if (rc == HA_ERR_END_OF_FILE) {
      break;
    }
    if (rc != 0) {
      ib::warn() << "VECINDEX: ha_rnd_next failed for aux table '" << aux_name
                 << "' rc=" << rc;
      seg->vid_pk_mapping.clear();
      return false;
    }

    std::vector<vec_pk_column_t> pk_cols(pk_fields);
    bool pk_ok = true;
    for (size_t i = 0; i < pk_fields; ++i) {
      Field *f = mysql_table->field[i];
      if (f == nullptr) {
        pk_ok = false;
        break;
      }

      vec_pk_column_t col;
      col.is_null = f->is_null();

      dict_field_t *df = clust_index->get_field(static_cast<ulint>(i));
      if (df != nullptr && df->col != nullptr) {
        col.mtype = df->col->mtype;
        col.prtype = df->col->prtype;
      }

      if (!col.is_null) {
        const uchar *ptr = f->data_ptr();
        const uint len = f->data_length();
        if (ptr == nullptr && len != 0) {
          pk_ok = false;
          break;
        }

        dtype_t dtype;
        dtype_set(&dtype, col.mtype, col.prtype, df->col->len);
        dfield_t dfield;
        dfield_set_type(&dfield, &dtype);
        /* store into buffer to match InnoDB key format */
        std::vector<byte> tmp(df->col->len + 16);
        byte *end = row_mysql_store_col_in_innobase_format(
            &dfield, tmp.data(), true, ptr, df->col->len,
            dict_table_is_comp(clust_index->table));
        const ulint stored_len =
            static_cast<ulint>(end - static_cast<byte *>(tmp.data()));
        col.data.assign(tmp.data(), tmp.data() + stored_len);
      }

      pk_cols[i] = std::move(col);
    }

    if (!pk_ok) {
      seg->vid_pk_mapping.clear();
      return false;
    }

    Field *faiss_field = mysql_table->field[pk_fields];
    if (faiss_field == nullptr || faiss_field->is_null()) {
      continue;
    }
    const ulonglong faiss_id_ull =
        static_cast<ulonglong>(faiss_field->val_int());
    if (faiss_id_ull == std::numeric_limits<ulonglong>::max()) {
      continue;
    }
    dberr_t cache_err = vec_insert_aux_cache(
        &seg->vid_pk_mapping, clust_index, static_cast<uint64_t>(faiss_id_ull),
        pk_cols);
    if (cache_err != DB_SUCCESS) {
      seg->vid_pk_mapping.clear();
      return false;
    }
  }

  h->ha_rnd_end();
  rnd_guard.h = nullptr;
  intern_close_table(mysql_table);
  table_guard.t = nullptr;
  vec_bitmap_mark_missing(&seg->vecindex_bitmap, aux_name,
                          seg->vid_pk_mapping.size(), thd);
  seg->vid_pk_mapping.ready = true;
  ib::warn() << "VECINDEX: loaded auxiliary PK cache for table '" << aux_name
             << "' rows=" << seg->vid_pk_mapping.size();

  /* Clean up the background transaction so the THD can be freed safely. */
  if (trx != nullptr &&
      trx->state.load(std::memory_order_relaxed) != TRX_STATE_NOT_STARTED) {
    trx_commit_for_mysql(trx);
  }
  if (allocated_trx) {
    trx_free_for_mysql(trx);
    thd_to_trx(thd) = nullptr;
  }
  return true;
}

static std::string vec_aux_pending_name(const std::string &prefix) {
  std::string name = prefix;
  name.append("_NEXT");
  return name;
}

bool vec_prepare_pending_mem_table(dict_index_t *index) {
  if (index == nullptr || index->vec_runtime == nullptr) {
    return false;
  }
  // delete pending name after use?
  THD *thd = current_thd;
  if (thd == nullptr) {
    ib::warn() << "VECINDEX: vec_prepare_pending_mem_table called without THD";
    return false;
  }

  vec_index_ctx_t *ctx = index->vec_runtime;

  std::unique_lock<std::shared_mutex> lk(ctx->mu);

  const std::string prefix =
      ctx->index_name_prefix.empty() ? vec_aux_prefix(index)
                                     : ctx->index_name_prefix;
  if (prefix.empty()) {
    ib::warn() << "VECINDEX: vec_prepare_pending_mem_table empty prefix";
    return false;
  }

  const std::string pending_name = vec_aux_pending_name(prefix);
  lk.unlock();
  ctx->pending_aux_name = pending_name;

  if (vec_aux_table_exists(pending_name)) {
    ib::warn() << "VECINDEX: pending aux table already exists " << pending_name;
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

  trx_t *&trx_slot = thd_to_trx(thd);
  trx_t *saved_trx = trx_slot;
  bool created = false;
  if (trx_slot == nullptr) {
    trx_slot = innobase_trx_allocate(thd);
    created = true;
  }
  trx_t *trx = trx_slot;
  if (trx == nullptr) {
    ib::warn() << "VECINDEX: failed to allocate trx for pending aux table";
    trx_slot = saved_trx;
    return false;
  }
  trx_start_if_not_started(trx, true, UT_LOCATION_HERE);

  dberr_t err = vec_aux_create_table(trx, index, pending_name);
  if (err == DB_SUCCESS) {
    index->fill_dd = true;
    if (index->table != nullptr) {
      err = vec_create_index_dd_tables(index->table);
    }
  }

  if (err == DB_SUCCESS) {
    err = trx_commit_for_mysql(trx);
  } else {
    trx_rollback_to_savepoint(trx, nullptr);
  }

  if (created) {
    trx_free_for_mysql(trx);
    trx_slot = saved_trx;
  }

  if (err != DB_SUCCESS) {
    ib::warn() << "VECINDEX: failed to prepare pending aux table "
               << pending_name << " err=" << err;
    if (adjust_autocommit) {
      thd->variables.option_bits = saved_options;
    }
    return false;
  }

  dict_table_t *hold =
      dd_table_open_on_name_in_mem(pending_name.c_str(), false);
  if (hold == nullptr) {
    hold = dd_table_open_on_name(thd, nullptr, pending_name.c_str(), false,
                                 DICT_ERR_IGNORE_NONE);
  }
  // if (hold != nullptr) {
  //   ib::warn() << "VECREF: hold pending aux dict table '" << pending_name
  //              << "' ref=" << hold->get_ref_count();
  // }

  lk.lock();
  ctx->pending_aux_name = pending_name;
  ctx->pending_aux_dict = hold;
  ctx->needs_aux_refresh.store(true);
  if (adjust_autocommit) {
    thd->variables.option_bits = saved_options;
  }
  return true;
}

bool vec_rotate_mem_index(trx_t *trx, dict_index_t *index) {
  ib::warn() << "VECINDEX: function::vec_rotate_mem_index() Rotating mem index.";
  if (trx == nullptr || index == nullptr || index->vec_runtime == nullptr) {
    ib::warn() << "VECINDEX: function::vec_rotate_mem_index() Invalid arguments.";
    return false;
  }

  /* Ensure caller is driving from a user thread; rotation performs DDL. */
  if (trx->state.load(std::memory_order_acquire) != TRX_STATE_ACTIVE) {
    ib::warn() << "VECINDEX: function::vec_rotate_mem_index() Transaction not active.";
    return false;
  }

  return vec_try_rotate_mem_index(trx, index, index->vec_runtime);
}

VecTaskManager &VecTaskManager::instance() {
  static VecTaskManager instance;
  return instance;
}

VecTaskManager::~VecTaskManager() { stop(); }

void VecTaskManager::ensure_started() {
  if (!running.load(std::memory_order_acquire)) {
    start();
  }
}

void VecTaskManager::start() {
  bool expected = false;
  if (!running.compare_exchange_strong(expected, true,
                                       std::memory_order_acq_rel)) {
    return;
  }
  worker_thread = std::thread(&VecTaskManager::worker_loop, this);
}

void VecTaskManager::stop() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    if (!running.load(std::memory_order_acquire)) {
      return;
    }
    running.store(false, std::memory_order_release);
  }
  cv.notify_all();
  if (worker_thread.joinable()) {
    worker_thread.join();
  }
}

void VecTaskManager::submit_task(dict_index_t *index) {
  if (index == nullptr || index->table == nullptr) {
    return;
  }
  ensure_started();

  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    const uint64_t key = static_cast<uint64_t>(index->id);
    if (pending_index_ids.find(key) != pending_index_ids.end()) {
      return;
    }
    tasks.push({index->table->id, index->id});
    pending_index_ids.insert(key);
  }
  ib::warn() << "VECINDEX: function::VecTaskManager::submit_task() Task submitted for index ID ";
  cv.notify_one();
}

void VecTaskManager::worker_loop() {
  bool thread_initialized = !my_thread_init();
  struct ThreadGuard {
    bool ok{false};
    ~ThreadGuard() {
      if (ok) {
        my_thread_end();
      }
    }
  } thread_guard{thread_initialized};

  if (!thread_initialized) {
    running.store(false, std::memory_order_release);
    cv.notify_all();
    return;
  }

  while (true) {
    VecBuildTask task{};
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      cv.wait(lock, [this] {
        return !tasks.empty() || !running.load(std::memory_order_acquire);
      });

      if (!running.load(std::memory_order_acquire) && tasks.empty()) {
        return;
      }

      task = tasks.front();
      tasks.pop();
      pending_index_ids.erase(static_cast<uint64_t>(task.index_id));
    }

    process_task(task);
  }
}

void VecTaskManager::process_task(VecBuildTask task) {
  if (task.table_id == 0 || task.index_id == 0) {
    return;
  }
  vec_rotate_mem_index_bg(task.table_id, task.index_id);
}

bool vec_recover_mutable_mem_index(dict_index_t *index, vec_index_ctx_t *ctx,
                                   THD *thd) {
  ib::warn() << "VECMETA: attempting mutable MEM recovery for index "
             << (index != nullptr && index->name ? index->name : "(null)");

  if (index == nullptr || ctx == nullptr || thd == nullptr ||
      index->table == nullptr) {
    return false;
  }

  vec_index_segment_t *seg = ctx->mutable_segment();
  if (seg == nullptr || seg->index == nullptr) {
    return false;
  }

  if (seg->index->ntotal() != 0) {
    ib::warn() << "VECMETA: mutable index already populated, skipping MEM "
               << "recovery for index "
               << (index->name ? index->name : "(null)");
    return true;
  }

  const std::string prefix =
      ctx->index_name_prefix.empty() ? vec_aux_prefix(index)
                                     : ctx->index_name_prefix;
  if (prefix.empty()) {
    return false;
  }

  const std::string mem_name = vec_aux_mem_name(prefix);
  if (!vec_aux_table_exists(mem_name)) {
    ib::warn() << "VECMETA: MEM table '" << mem_name
               << "' not found, skip mutable recovery.";
    return true;
  }

  if (seg->aux_table_name.empty()) {
    seg->aux_table_name = mem_name;
  }

  dict_index_t *clust_index = index->table->first_index();
  if (clust_index == nullptr) {
    return false;
  }

  const ulint pk_fields = dict_index_get_n_unique(clust_index);
  if (pk_fields == 0) {
    return false;
  }

  const dict_field_t *vec_field = index->get_field(0);
  if (vec_field == nullptr || vec_field->col == nullptr) {
    return false;
  }

  const size_t vec_col_no = dict_col_get_no(vec_field->col);
  const size_t dim = ctx->params.dim;
  if (dim == 0) {
    return false;
  }

  auto split_name = [](const std::string &full, std::string *db,
                       std::string *tbl) -> bool {
    if (full.empty() || db == nullptr || tbl == nullptr) {
      return false;
    }
    const auto slash = full.find('/');
    if (slash == std::string::npos || slash == 0 || slash + 1 >= full.size()) {
      return false;
    }
    *db = full.substr(0, slash);
    *tbl = full.substr(slash + 1);
    return !(db->empty() || tbl->empty());
  };

  std::string aux_db, aux_tbl;
  if (!split_name(mem_name, &aux_db, &aux_tbl)) {
    return false;
  }

  std::string base_db, base_tbl;
  if (index->table->name.m_name == nullptr ||
      !split_name(index->table->name.m_name, &base_db, &base_tbl)) {
    return false;
  }

  MDL_request aux_mdl;
  MDL_request base_mdl;
  MDL_REQUEST_INIT(&aux_mdl, MDL_key::TABLE, aux_db.c_str(), aux_tbl.c_str(),
                   MDL_SHARED_READ, MDL_EXPLICIT);
  MDL_REQUEST_INIT(&base_mdl, MDL_key::TABLE, base_db.c_str(),
                   base_tbl.c_str(), MDL_SHARED_READ, MDL_EXPLICIT);

  if (thd->mdl_context.acquire_lock(&aux_mdl, 0)) {
    ib::warn() << "VECMETA: failed to acquire MDL on MEM table '" << mem_name
               << "'";
    return false;
  }

  if (thd->mdl_context.acquire_lock(&base_mdl, 0)) {
    ib::warn() << "VECMETA: failed to acquire MDL on base table '"
               << index->table->name.m_name << "'";
    thd->mdl_context.release_lock(aux_mdl.ticket);
    return false;
  }

  struct MdlGuard {
    THD *thd{nullptr};
    MDL_request *req{nullptr};
    ~MdlGuard() {
      if (thd != nullptr && req != nullptr && req->ticket != nullptr) {
        thd->mdl_context.release_lock(req->ticket);
        req->ticket = nullptr;
      }
    }
  };

  MdlGuard aux_guard{thd, &aux_mdl};
  MdlGuard base_guard{thd, &base_mdl};

  dd::cache::Dictionary_client *client = dd::get_dd_client(thd);
  dd::cache::Dictionary_client::Auto_releaser releaser(client);

  const dd::Table *dd_aux = nullptr;
  const dd::Table *dd_base = nullptr;
  if (client->acquire<dd::Table>(aux_db.c_str(), aux_tbl.c_str(), &dd_aux) ||
      dd_aux == nullptr) {
    ib::warn() << "VECMETA: failed to acquire DD for MEM table '" << mem_name
               << "'";
    return false;
  }

  if (client->acquire<dd::Table>(base_db.c_str(), base_tbl.c_str(), &dd_base) ||
      dd_base == nullptr) {
    ib::warn() << "VECMETA: failed to acquire DD for base table '"
               << index->table->name.m_name << "'";
    return false;
  }

  char aux_path[FN_REFLEN + 1]{};
  char base_path[FN_REFLEN + 1]{};
  bool aux_trunc = false;
  bool base_trunc = false;
  size_t aux_path_len =
      build_table_filename(aux_path, sizeof(aux_path) - 1 - reg_ext_length,
                           aux_db.c_str(), aux_tbl.c_str(), "", 0, &aux_trunc);
  size_t base_path_len = build_table_filename(
      base_path, sizeof(base_path) - 1 - reg_ext_length, base_db.c_str(),
      base_tbl.c_str(), "", 0, &base_trunc);
  if (aux_path_len == 0 || aux_trunc || base_path_len == 0 || base_trunc) {
    ib::warn() << "VECMETA: failed to build table path for MEM or base table "
               << "during recovery.";
    return false;
  }

  TABLE *aux_table =
      open_table_uncached(thd, aux_path, aux_db.c_str(), aux_tbl.c_str(),
                          false, true, *dd_aux);
  TABLE *base_table =
      open_table_uncached(thd, base_path, base_db.c_str(), base_tbl.c_str(),
                          false, true, *dd_base);

  if (aux_table == nullptr || aux_table->file == nullptr || base_table == nullptr ||
      base_table->file == nullptr) {
    ib::warn() << "VECMETA: failed to open MEM or base table for recovery.";
    if (aux_table != nullptr) {
      intern_close_table(aux_table);
    }
    if (base_table != nullptr) {
      intern_close_table(base_table);
    }
    return false;
  }

  struct TableGuard {
    TABLE *t{nullptr};
    ~TableGuard() {
      if (t != nullptr) {
        intern_close_table(t);
        t = nullptr;
      }
    }
  };

  TableGuard aux_table_guard{aux_table};
  TableGuard base_table_guard{base_table};

  trx_t *trx = thd_to_trx(thd);
  bool allocated_trx = false;
  if (trx == nullptr) {
    trx = innobase_trx_allocate(thd);
    if (trx == nullptr) {
      ib::warn() << "VECMETA: failed to allocate trx for MEM recovery";
      return false;
    }
    thd_to_trx(thd) = trx;
    allocated_trx = true;
  }
  trx_start_if_not_started(trx, true, UT_LOCATION_HERE);

  struct TrxCleanupGuard {
    THD *thd{nullptr};
    trx_t *trx{nullptr};
    bool owned{false};
    bool committed{false};
    ~TrxCleanupGuard() {
      if (trx == nullptr) {
        return;
      }
      if (!committed && owned &&
          trx->state.load(std::memory_order_relaxed) != TRX_STATE_NOT_STARTED) {
        trx_rollback_to_savepoint(trx, nullptr);
      }
      if (owned) {
        trx_free_for_mysql(trx);
        if (thd != nullptr) {
          thd_to_trx(thd) = nullptr;
        }
      }
    }
  } trx_cleanup{thd, trx, allocated_trx, false};

  struct TrxDepthGuard {
    trx_t *trx{nullptr};
    bool entered{false};
    ~TrxDepthGuard() {
      if (entered && trx != nullptr) {
        TrxInInnoDB::end_stmt(trx);
      }
    }
  } depth_guard{trx, false};
  TrxInInnoDB::begin_stmt(trx);
  depth_guard.entered = true;

  handler *aux_h = aux_table->file;
  handler *base_h = base_table->file;
  if (aux_table->read_set != nullptr) {
    bitmap_set_all(aux_table->read_set);
  }
  if (base_table->read_set != nullptr) {
    bitmap_set_all(base_table->read_set);
  }

  if (pk_fields >= aux_table->s->fields) {
    ib::warn() << "VECMETA: MEM table layout mismatch for '" << mem_name << "'";
    return false;
  }

  if (aux_h->ha_rnd_init(true)) {
    ib::warn() << "VECMETA: ha_rnd_init failed for MEM table '" << mem_name
               << "'";
    return false;
  }

  struct RndEndGuard {
    handler *h{nullptr};
    ~RndEndGuard() {
      if (h != nullptr) {
        h->ha_rnd_end();
      }
    }
  } aux_rnd{aux_h};

  struct AuxEntry {
    std::vector<vec_pk_column_t> pk_cols;
    uint64_t old_id{0};
    uint64_t new_id{std::numeric_limits<uint64_t>::max()};
  };

  std::vector<AuxEntry> entries;
  std::unordered_map<std::string, size_t> pk_to_entry;

  while (true) {
    int rc = aux_h->ha_rnd_next(aux_table->record[0]);
    if (rc == HA_ERR_END_OF_FILE) {
      break;
    }
    if (rc != 0) {
      ib::warn() << "VECMETA: ha_rnd_next failed for MEM table '" << mem_name
                 << "' rc=" << rc;
      return false;
    }

    AuxEntry entry{};
    bool pk_ok = vec_extract_pk_columns(
        aux_table, clust_index, pk_fields,
        [aux_table](size_t idx) {
          return (idx < static_cast<size_t>(aux_table->s->fields))
                     ? aux_table->field[idx]
                     : nullptr;
        },
        entry.pk_cols);
    if (!pk_ok) {
      ib::warn() << "VECMETA: failed to decode PK from MEM row";
      return false;
    }

    entry.old_id = 0;

    std::string key = vec_pack_pk_key(entry.pk_cols, pk_fields);
    if (key.empty()) {
      continue;
    }

    if (pk_to_entry.find(key) != pk_to_entry.end()) {
      continue;
    }

    pk_to_entry.emplace(std::move(key), entries.size());
    entries.push_back(std::move(entry));
  }

  if (entries.empty()) {
    ib::warn() << "VECMETA: MEM table '" << mem_name
               << "' empty; nothing to recover.";
    if (trx != nullptr &&
        trx->state.load(std::memory_order_relaxed) != TRX_STATE_NOT_STARTED) {
      trx_commit_for_mysql(trx);
    }
    trx_cleanup.committed = true;
    return true;
  }

  if (base_h->ha_rnd_init(true)) {
    ib::warn() << "VECMETA: ha_rnd_init failed for base table '"
               << index->table->name.m_name << "'";
    return false;
  }
  RndEndGuard base_rnd{base_h};

  auto base_field_resolver = [&](size_t idx) -> Field * {
    dict_field_t *df = clust_index->get_field(static_cast<ulint>(idx));
    if (df == nullptr || df->col == nullptr) {
      return nullptr;
    }
    const ulint col_no = dict_col_get_no(df->col);
    if (col_no >= base_table->s->fields) {
      return nullptr;
    }
    return base_table->field[col_no];
  };

  std::vector<float> xb;
  std::vector<int64_t> ids;
  xb.reserve(entries.size() * dim);
  ids.reserve(entries.size());

  while (true) {
    int rc = base_h->ha_rnd_next(base_table->record[0]);
    if (rc == HA_ERR_END_OF_FILE) {
      break;
    }
    if (rc != 0) {
      ib::warn() << "VECMETA: ha_rnd_next failed for base table '"
                 << index->table->name.m_name << "' rc=" << rc;
      return false;
    }

    std::vector<vec_pk_column_t> pk_cols;
    if (!vec_extract_pk_columns(base_table, clust_index, pk_fields,
                                base_field_resolver, pk_cols)) {
      ib::warn() << "VECMETA: failed to extract PK from base row";
      return false;
    }

    std::string key = vec_pack_pk_key(pk_cols, pk_fields);
    auto it = pk_to_entry.find(key);
    if (it == pk_to_entry.end()) {
      continue;
    }

    AuxEntry &entry = entries[it->second];
    if (entry.new_id != std::numeric_limits<uint64_t>::max()) {
      continue;
    }

    std::vector<float> vec_values;
    if (!vec_extract_vector_field(base_table, vec_col_no, dim, vec_values)) {
      ib::warn() << "VECMETA: invalid vector payload during MEM recovery";
      continue;
    }

    const uint64_t new_id = static_cast<uint64_t>(ids.size());
    entry.new_id = new_id;
    ids.push_back(static_cast<int64_t>(new_id));
    xb.insert(xb.end(), vec_values.begin(), vec_values.end());

    dberr_t cache_err = vec_insert_aux_cache(
        &seg->vid_pk_mapping, clust_index, new_id, entry.pk_cols);
    seg->vecindex_bitmap.ensure_size(seg->vid_pk_mapping.size());
    if (cache_err != DB_SUCCESS) {
      ib::warn() << "VECMETA: failed to insert pk cache during MEM recovery "
                 << "for id=" << new_id;
      return false;
    }
  }

  if (ids.empty()) {
    ib::warn() << "VECMETA: no matching base rows found for MEM recovery on '"
               << index->table->name.m_name << "'";
    if (trx != nullptr &&
        trx->state.load(std::memory_order_relaxed) != TRX_STATE_NOT_STARTED) {
      trx_commit_for_mysql(trx);
    }
    trx_cleanup.committed = true;
    return true;
  }

  {
    std::lock_guard<std::shared_mutex> lk(ctx->mu);
    seg->index->add(ids.size(), xb.data(), ids.data());
  }

  seg->vecindex_bitmap.clear();
  seg->vecindex_bitmap.ensure_size(seg->vid_pk_mapping.size());
  seg->vid_pk_mapping.ready = true;
  ctx->id_alloc.next = static_cast<uint64_t>(seg->index->ntotal());
  ctx->id_alloc.inited = true;

  // MEM recovery rebuilds contiguous faiss_id, so disable mem_diff remap.
  seg->vid_pk_mapping.is_mem_diff = false;
  seg->vid_pk_mapping.mem_diff.clear();
#if 0
  // Align mem index if there are deletions making the mapping inconsistent.
  {
    const uint64_t invalid_id = std::numeric_limits<uint64_t>::max();
    uint64_t max_old_id = 0;
    bool have_old_id = false;
    for (const auto &entry : entries) {
      if (entry.new_id == invalid_id) {
        continue;
      }
      max_old_id = std::max(max_old_id, entry.old_id);
      have_old_id = true;
    }
    const uint64_t mapping_size =
        static_cast<uint64_t>(seg->vid_pk_mapping.size());
    if (have_old_id && max_old_id >= mapping_size) {
      if (max_old_id >=
          static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        /* Avoid oversized allocations; skip mem_diff in this case. */
      } else {
        ib::warn()<< "VECMETA: aligning mem index for table '"
                   << index->table->name.m_name << "' old_id_max="
                   << max_old_id << " mapping_size=" << mapping_size;
        seg->vid_pk_mapping.is_mem_diff = true;
        seg->vid_pk_mapping.mem_diff.assign(
            static_cast<size_t>(max_old_id + 1), -1);
      }
      for (const auto &entry : entries) {
        if (entry.new_id == invalid_id) {
          continue;
        }
        if (!seg->vid_pk_mapping.is_mem_diff ||
            entry.old_id >= seg->vid_pk_mapping.mem_diff.size()) {
          continue;
        }
        const int64_t diff =
            static_cast<int64_t>(entry.old_id) -
            static_cast<int64_t>(entry.new_id);
        seg->vid_pk_mapping.mem_diff[entry.old_id] = diff;
      }
    }
  }
#endif

  if (trx != nullptr &&
      trx->state.load(std::memory_order_relaxed) != TRX_STATE_NOT_STARTED) {
    trx_commit_for_mysql(trx);
  }
  trx_cleanup.committed = true;

  ib::warn() << "VECMETA: recovered " << ids.size()
             << " vectors into mutable MEM index for table '"
             << index->table->name.m_name << "'";
  return true;
}

class VecMetaLoader {
 public:
  static VecMetaLoader &instance() {
    static VecMetaLoader loader;
    return loader;
  }

  ~VecMetaLoader() { stop(); }

  void schedule(dict_index_t *index) {
    if (index == nullptr || index->vec_runtime == nullptr) {
      return;
    }
    vec_index_ctx_t *ctx = index->vec_runtime;
    if (ctx->bootstrap_load_submitted.exchange(true,
                                               std::memory_order_acq_rel)) {
      return;
    }

    ensure_started();
    {
      std::lock_guard<std::mutex> lock(mu);
      const uint64_t key = static_cast<uint64_t>(index->id);
      if (pending.count(key) != 0) {
        return;
      }
      tasks.push(index);
      pending.insert(key);
    }
    cv.notify_one();
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mu);
      if (!running.load(std::memory_order_acquire)) {
        return;
      }
      running.store(false, std::memory_order_release);
    }
    cv.notify_all();
    if (worker.joinable()) {
      worker.join();
    }
  }

 private:
  VecMetaLoader() = default;
  VecMetaLoader(const VecMetaLoader &) = delete;
  VecMetaLoader &operator=(const VecMetaLoader &) = delete;

  void ensure_started() {
    bool expected = false;
    if (!running.compare_exchange_strong(expected, true,
                                         std::memory_order_acq_rel)) {
      return;
    }
    worker = std::thread(&VecMetaLoader::worker_loop, this);
  }

  void worker_loop() {
    bool thread_initialized = !my_thread_init();
    struct ThreadGuard {
      bool ok{false};
      ~ThreadGuard() {
        if (ok) {
          my_thread_end();
        }
      }
    } guard{thread_initialized};

    if (!thread_initialized) {
      running.store(false, std::memory_order_release);
      cv.notify_all();
      return;
    }

    while (true) {
      dict_index_t *index = nullptr;
      {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [this] {
          return !tasks.empty() || !running.load(std::memory_order_acquire);
        });

        if (!running.load(std::memory_order_acquire) && tasks.empty()) {
          return;
        }
        index = tasks.front();
        tasks.pop();
        if (index != nullptr) {
          pending.erase(static_cast<uint64_t>(index->id));
        }
      }

      process(index);
    }
  }

  void process(dict_index_t *index) {
    ib::warn() << "VECMETA: starting bootstrap load for index "
               << (index->name ? index->name : "(null)");

    vec_index_ctx_t *ctx = index != nullptr ? index->vec_runtime : nullptr;
    auto mark_ready = [ctx]() {
      if (ctx != nullptr) {
        ctx->bootstrap_state.store(VecBootstrapState::READY,
                                   std::memory_order_release);
        ctx->bootstrap_loaded.store(true, std::memory_order_release);
      }
    };
    auto mark_failed = [ctx]() {
      if (ctx != nullptr) {
        ctx->bootstrap_state.store(VecBootstrapState::FAILED,
                                   std::memory_order_release);
        ctx->bootstrap_loaded.store(false, std::memory_order_release);
        ctx->bootstrap_load_submitted.store(false, std::memory_order_release);
      }
    };

    if (index == nullptr || ctx == nullptr || index->vec_params == nullptr) {
      mark_failed();
      ib::warn() << "VECMETA: invalid index for bootstrap load.";
      return;
    }

    ib::warn() << "VECMETA: processing bootstrap load for index "
               << (index->name ? index->name : "(null)");
    std::string meta_path;
    VecBgThdGuard guard;
    if (!guard.create()) {
      mark_failed();
      ib::warn() << "VECMETA: failed to create THD for aux cache preload";
      return;
    }

    VecMetaHeader header{};
    std::vector<VecSegmentEntry> entries;
    bool meta_ok = false;
    bool meta_expected = false;
    if (vec_meta_path_for_index(index, &meta_path)) {
      meta_expected = true;
      if (my_access(meta_path.c_str(), F_OK) != 0) {
        meta_expected = false;
      } else if (vec_meta_read_all(meta_path, &header, &entries)) {
        VecMetaHeader expected = vec_meta_make_header(index, ctx->params);
        auto header_matches = [](const VecMetaHeader &lhs,
                                 const VecMetaHeader &rhs) {
          return lhs.magic == rhs.magic && lhs.version == rhs.version &&
                 lhs.index_id == rhs.index_id &&
                 lhs.dimension == rhs.dimension &&
                 lhs.index_type == rhs.index_type &&
                 lhs.metric_type == rhs.metric_type;
        };

        if (!header_matches(header, expected)) {
          meta_expected = true;
          ib::warn() << "VECMETA: header mismatch for index "
                     << (index->name ? index->name : "(null)")
                     << ", skip meta segments.";
        } else {
          meta_ok = true;
        }
      } else {
        ib::warn() << "VECMETA: failed to read meta file '" << meta_path << "'";
      }
    } else {
      ib::warn() << "VECMETA: no meta path available for index "
                 << (index->name ? index->name : "(null)")
                 << "; attempting MEM recovery only.";
    }

    bool ok = true;
    if (meta_expected && !meta_ok) {
      ok = false;
    }

    const std::string base_dir = vec_meta_dirname(meta_path);
    const std::string prefix =
        ctx->index_name_prefix.empty() ? vec_aux_prefix(index)
                                       : ctx->index_name_prefix;

    if (meta_ok) {
      ib::warn() << "VECMETA: loading " << entries.size()
                 << " segments for index "
                 << (index->name ? index->name : "(null)");

      for (const auto &entry : entries) {
        vec_index_segment_t *inserted = nullptr;
        ib::warn() << "VECMETA: processing segment " << entry.seg_id
                   << " with state " << static_cast<int>(entry.state)
                   << " and file name '" << entry.file_name << "'";
        if (entry.state != static_cast<uint8_t>(VecSegmentState::Committed)) {
          continue;
        }
        const uint32_t seg_id = static_cast<uint32_t>(entry.seg_id);

        {
          std::lock_guard<std::shared_mutex> lk(ctx->mu);
          const bool exists = std::any_of(
              ctx->segments.begin(), ctx->segments.end(),
              [seg_id](const vec_index_segment_t &s) {
                return s.vecindex_id == seg_id && s.immutable;
              });
          if (exists) {
            continue;
          }
        }

        std::string seg_path = vec_meta_join(base_dir, entry.file_name);
        if (seg_path.empty()) {
          ib::warn() << "VECMETA: empty segment path for seg_id=" << seg_id;
          continue;
        }

        auto target = ctx->params.backend == BackendType::Faiss
                          ? vec_make_faiss_index(ctx->params)
                          : vec_make_hnswlib_index(ctx->params);
        if (!target) {
          ib::warn() << "VECMETA: unable to allocate index for seg_id="
                     << seg_id;
          continue;
        }
        
        ib::warn() << "VECMETA: loading segment " << seg_id;
        try {
          target->load(seg_path);
        } catch (const std::exception &e) {
          ok = false;
          ib::warn() << "VECMETA: failed to load segment file '" << seg_path
                     << "' error=" << e.what();
          continue;
        }

        vec_index_segment_t loaded{};
        loaded.index = std::move(target);
        loaded.aux_table_name =
            prefix.empty() ? std::string{} : vec_aux_segment_name(prefix, seg_id);
        loaded.index_file_name = seg_path;
        loaded.vecindex_id = seg_id;
        loaded.immutable = true;

        const bool expect_pk_map = vec_meta_segment_has_pk_mapping(entry);
        const std::string map_path = vec_vid_pk_mapping_path(seg_path);
        if (!map_path.empty()) {
          const bool map_ok =
              vec_vid_pk_mapping_load(map_path, &loaded.vid_pk_mapping);
          if (map_ok) {
            vec_bitmap_mark_missing(&loaded.vecindex_bitmap,
                                    loaded.aux_table_name,
                                    loaded.vid_pk_mapping.size(),
                                    guard.thd());
            ib::warn() << "VECMETA: restored PK mapping from '" << map_path
                       << "' rows=" << loaded.vid_pk_mapping.size();
            loaded.vid_pk_mapping.ready = true;
          } else if (expect_pk_map) {
            ib::warn() << "VECMETA: pk mapping flagged but not readable at '"
                       << map_path << "'";
          }
        }

        std::lock_guard<std::shared_mutex> lk(ctx->mu);
        const bool dup = std::any_of(
            ctx->segments.begin(), ctx->segments.end(),
            [seg_id](const vec_index_segment_t &s) {
              return s.vecindex_id == seg_id && s.immutable;
            });
        if (!dup) {
          ctx->segments.push_back(std::move(loaded));
          ctx->max_vecindex_id =
              std::max<uint32_t>(ctx->max_vecindex_id, seg_id);
          inserted = &ctx->segments.back();
        }

        ib::warn() << "VECMETA: loaded segment " << seg_id
                   << " for index "
                   << (index->name ? index->name : "(null)");

        if (inserted != nullptr && guard.thd() != nullptr &&
            !inserted->vid_pk_mapping.ready) {
              //TODO: May need a flag to resave PK mapping to file?
          if (!vec_load_aux_cache_for_segment(index, ctx, inserted, guard.thd())) {
            ok = false;
          }
        }
      }
    }

    if (guard.thd() != nullptr &&
        !vec_recover_mutable_mem_index(index, ctx, guard.thd())) {
      ib::warn() << "VECMETA: MEM recovery failed for index "
                 << (index->name ? index->name : "(null)");
      ok = false;
    }
    if (guard.thd() == nullptr) {
      ok = false;
    }

    if (ok) {
      mark_ready();
    } else {
      mark_failed();
    }
  }

  std::queue<dict_index_t *> tasks;
  std::set<uint64_t> pending;
  std::mutex mu;
  std::condition_variable cv;
  std::thread worker;
  std::atomic<bool> running{false};
};

void vec_schedule_bootstrap_load(dict_index_t *index) {
  if (index == nullptr || index->vec_runtime == nullptr) {
    return;
  }
  vec_index_ctx_t *ctx = index->vec_runtime;
  VecBootstrapState state =
      ctx->bootstrap_state.load(std::memory_order_acquire);
  if (state == VecBootstrapState::READY ||
      state == VecBootstrapState::LOADING) {
    return;
  }

  if (state == VecBootstrapState::FAILED) {
    ctx->bootstrap_load_submitted.store(false, std::memory_order_release);
  }

  VecBootstrapState expected = state;
  if (!ctx->bootstrap_state.compare_exchange_strong(
          expected, VecBootstrapState::LOADING,
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    return;
  }
  ctx->bootstrap_loaded.store(false, std::memory_order_release);
  VecMetaLoader::instance().schedule(index);
}
