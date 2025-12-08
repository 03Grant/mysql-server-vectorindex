// vec_tasks.cc
#include "vec_tasks.h"

#include <algorithm>
#include <exception>
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

namespace {

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

bool vec_try_rotate_mem_index(trx_t *trx, dict_index_t *index,
                              vec_index_ctx_t *ctx) {
  ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() 1 Trying to rotate mem index.";
  if (trx == nullptr || index == nullptr || ctx == nullptr) {
    ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() Invalid arguments.";
    return false;
  }

  ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() 2 Trying to rotate mem index.";
  std::unique_lock<std::mutex> lk(ctx->mu);
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

  if (!ctx->segments.empty()) {
    ctx->segments.erase(ctx->segments.begin());
  }
  ctx->segments.insert(ctx->segments.begin(), std::move(fresh));
  ctx->segments.push_back(std::move(staging));
  ctx->max_vecindex_id = std::max(ctx->max_vecindex_id, seg_id);
  if (ctx->pending_aux_dict != nullptr) {
    dd_table_close(ctx->pending_aux_dict, nullptr, nullptr, false);
    ctx->pending_aux_dict = nullptr;
  }
  ctx->pending_aux_name.clear();
  ctx->needs_aux_refresh.store(true);
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
    if (trx_ != nullptr) {
      trx_free_for_mysql(trx_);
      trx_ = nullptr;
      if (thd_ != nullptr) {
        thd_to_trx(thd_) = nullptr;
      }
    }
    if (thd_ != nullptr) {
      destroy_thd(thd_);
      thd_ = nullptr;
      current_thd = prev_thd_;
    }
  }

  THD *thd_{nullptr};
  trx_t *trx_{nullptr};
  THD *prev_thd_{nullptr};
};

bool vec_rotate_mem_index_bg(dict_index_t *index, vec_index_ctx_t *ctx) {
  ib::warn() << "VECINDEX: function::vec_rotate_mem_index_bg() Rotating mem index in bg.";
  if (index == nullptr || ctx == nullptr) {
    ib::warn() << "VECINDEX: function::vec_rotate_mem_index_bg() Invalid arguments.";
    return false;
  }

  VecBgThdGuard guard;
  if (!guard.create()) {
    ib::warn() << "VECINDEX: function::vec_rotate_mem_index_bg() Failed to create background thread.";
    return false;
  }
  if (!guard.start_trx()) {
    ib::warn() << "VECINDEX: function::vec_rotate_mem_index_bg() Failed to start transaction.";
    return false;
  }

  bool rotated = vec_try_rotate_mem_index(guard.trx(), index, ctx);
  if (rotated) {
    rotated = guard.commit();
  } else {
    guard.rollback();
  }
  ib::warn() << "VECINDEX: function::vec_rotate_mem_index_bg() Rotation "
             << (rotated ? "succeeded." : "failed.");
  return rotated;
}

void vec_bg_build_task(dict_index_t *index, vec_index_ctx_t *ctx,
                       vec_index_segment_t *seg) {
  ib::warn() << "VECINDEX: function::vec_bg_build_task() Building index segment in bg.";
  if (index == nullptr || ctx == nullptr || seg == nullptr) {
    ib::warn() << "VECINDEX: function::vec_bg_build_task() Invalid arguments.";
    return;
  }
  ctx->build_in_progress = true;
  auto clear_build_flag = [&]() {
    ctx->build_in_progress = false;
  };

  std::vector<float> xb;
  std::vector<int64_t> ids;
  const size_t dim = ctx->params.dim;
  if (dim == 0 || !vec_dump_segment(seg, dim, xb, ids)) {
    clear_build_flag();
    ib::warn() << "VECINDEX: function::vec_bg_build_task() Failed to dump segment data.";
    return;
  }

  std::string index_path;
  if (!vec_resolve_index_path(seg->aux_table_name, &index_path)) {
    clear_build_flag();
    ib::warn() << "VECINDEX: function::vec_bg_build_task() Failed to resolve index path.";
    return;
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
    return;
  }

  target->train(ids.size(), xb.data());
  target->add(ids.size(), xb.data(), ids.data());

  target->save(index_path);
  seg->index_file_name = index_path;
  seg->index = std::move(target);

  if (meta_ready) {
    meta_entry.state = static_cast<uint8_t>(VecSegmentState::Committed);
    if (!meta_file.overwrite(meta_offset, &meta_entry)) {
      ib::warn() << "VECMETA: failed to commit meta entry for seg "
                 << seg->vecindex_id;
    }
  }

  clear_build_flag();
  ib::warn() << "VECINDEX: function::vec_bg_build_task() Finished building index segment.";
}

}  // namespace

bool vec_load_aux_cache_for_segment(dict_index_t *vec_index,
                                    vec_index_ctx_t *ctx,
                                    vec_index_segment_t *seg, THD *thd) {
  if (vec_index == nullptr || vec_index->table == nullptr || ctx == nullptr ||
      seg == nullptr || thd == nullptr) {
    return false;
  }

  if (seg->aux_cache.ready) {
    return true;
  }

  dict_index_t *clust_index = vec_index->table->first_index();
  if (clust_index == nullptr) {
    return false;
  }

  const size_t pk_fields = dict_index_get_n_unique(clust_index);
  if (pk_fields == 0) {
    return false;
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

  seg->aux_cache.clear();

  while (true) {
    int rc = h->ha_rnd_next(mysql_table->record[0]);
    if (rc == HA_ERR_END_OF_FILE) {
      break;
    }
    if (rc != 0) {
      ib::warn() << "VECINDEX: ha_rnd_next failed for aux table '" << aux_name
                 << "' rc=" << rc;
      seg->aux_cache.clear();
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
      seg->aux_cache.clear();
      return false;
    }

    Field *faiss_field = mysql_table->field[pk_fields];
    if (faiss_field == nullptr || faiss_field->is_null()) {
      continue;
    }
    const ulonglong faiss_id_ull =
        static_cast<ulonglong>(faiss_field->val_int());
    dberr_t cache_err = vec_insert_aux_cache(
        &seg->aux_cache, clust_index, static_cast<uint64_t>(faiss_id_ull),
        pk_cols);
    if (cache_err != DB_SUCCESS) {
      seg->aux_cache.clear();
      return false;
    }
  }

  h->ha_rnd_end();
  rnd_guard.h = nullptr;
  intern_close_table(mysql_table);
  table_guard.t = nullptr;
  seg->aux_cache.ready = true;
  ib::warn() << "VECINDEX: loaded auxiliary PK cache for table '" << aux_name
             << "' rows=" << seg->aux_cache.size();

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

  std::unique_lock<std::mutex> lk(ctx->mu);

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
  if (index == nullptr) {
    return;
  }
  ensure_started();

  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    const uint64_t key = static_cast<uint64_t>(index->id);
    if (pending_index_ids.find(key) != pending_index_ids.end()) {
      return;
    }
    tasks.push({index});
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
      if (task.index != nullptr) {
        pending_index_ids.erase(static_cast<uint64_t>(task.index->id));
      }
    }

    process_task(task);
  }
}

void VecTaskManager::process_task(VecBuildTask task) {
  dict_index_t *index = task.index;
  if (index == nullptr || index->vec_runtime == nullptr) {
    return;
  }

  vec_index_ctx_t *ctx = index->vec_runtime;
  auto clear_pending = [ctx]() {
    if (ctx != nullptr) {
      ctx->is_rotation_pending.store(false);
    }
  };

  if (!vec_rotate_mem_index_bg(index, ctx)) {
    clear_pending();
    return;
  }

  vec_index_segment_t *staging_seg = nullptr;
  {
    std::lock_guard<std::mutex> lk(ctx->mu);
    if (!ctx->segments.empty()) {
      staging_seg = &ctx->segments.back();
    }
  }

  if (staging_seg != nullptr && staging_seg->immutable) {
    vec_bg_build_task(index, ctx, staging_seg);
  }

  clear_pending();
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

    if (index == nullptr || index->vec_runtime == nullptr ||
        index->vec_params == nullptr) {
      if (index != nullptr && index->vec_runtime != nullptr) {
        index->vec_runtime->bootstrap_loaded.store(
            true, std::memory_order_release);
      }
      ib::warn() << "VECMETA: invalid index for bootstrap load.";
      return;
    }

    ib::warn() << "VECMETA: processing bootstrap load for index "
               << (index->name ? index->name : "(null)");
    vec_index_ctx_t *ctx = index->vec_runtime;
    std::string meta_path;
    if (!vec_meta_path_for_index(index, &meta_path)) {
      ctx->bootstrap_loaded.store(true, std::memory_order_release);
      ib::warn() << "VECMETA: failed to get meta path for index "
                 << (index->name ? index->name : "(null)");
      return;
    }

    VecBgThdGuard guard;
    if (!guard.create()) {
      ctx->bootstrap_loaded.store(true, std::memory_order_release);
      ib::warn() << "VECMETA: failed to create THD for aux cache preload";
      return;
    }

    VecMetaHeader header{};
    std::vector<VecSegmentEntry> entries;
    if (!vec_meta_read_all(meta_path, &header, &entries)) {
      ctx->bootstrap_loaded.store(true, std::memory_order_release);
      ib::warn() << "VECMETA: failed to read meta file '" << meta_path
                 << "'";
      return;
    }

    VecMetaHeader expected = vec_meta_make_header(index, ctx->params);
    auto header_matches = [](const VecMetaHeader &lhs,
                             const VecMetaHeader &rhs) {
      return lhs.magic == rhs.magic && lhs.version == rhs.version &&
             lhs.index_id == rhs.index_id && lhs.dimension == rhs.dimension &&
             lhs.index_type == rhs.index_type &&
             lhs.metric_type == rhs.metric_type;
    };

    if (!header_matches(header, expected)) {
      ib::warn() << "VECMETA: header mismatch for index "
                 << (index->name ? index->name : "(null)")
                 << ", skip bootstrap load.";
      ctx->bootstrap_loaded.store(true, std::memory_order_release);
      return;
    }

    const std::string base_dir = vec_meta_dirname(meta_path);
    const std::string prefix =
        ctx->index_name_prefix.empty() ? vec_aux_prefix(index)
                                       : ctx->index_name_prefix;

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
        std::lock_guard<std::mutex> lk(ctx->mu);
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

      std::lock_guard<std::mutex> lk(ctx->mu);
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

      if (inserted != nullptr && guard.thd() != nullptr) {
        vec_load_aux_cache_for_segment(index, ctx, inserted, guard.thd());
      }
    }

    ctx->bootstrap_loaded.store(true, std::memory_order_release);
  }

  std::queue<dict_index_t *> tasks;
  std::set<uint64_t> pending;
  std::mutex mu;
  std::condition_variable cv;
  std::thread worker;
  std::atomic<bool> running{false};
};

void vec_schedule_bootstrap_load(dict_index_t *index) {
  VecMetaLoader::instance().schedule(index);
}
