// vec_tasks.cc
#include "vec_tasks.h"

#include <algorithm>
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
#include "sql/sql_table.h"
#include "sql/sql_class.h"
#include "sql/sql_lex.h"
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
#include "vec_params.h"

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

void vec_bg_build_task(vec_index_ctx_t *ctx, vec_index_segment_t *seg) {
  ib::warn() << "VECINDEX: function::vec_bg_build_task() Building index segment in bg.";
  if (ctx == nullptr || seg == nullptr) {
    ib::warn() << "VECINDEX: function::vec_bg_build_task() Invalid arguments.";
    return;
  }
  ctx->build_in_progress = true;
  std::vector<float> xb;
  std::vector<int64_t> ids;
  const size_t dim = ctx->params.dim;
  if (dim == 0 || !vec_dump_segment(seg, dim, xb, ids)) {
    ctx->build_in_progress = false;
    ib::warn() << "VECINDEX: function::vec_bg_build_task() Failed to dump segment data.";
    return;
  }

  auto target = ctx->params.backend == BackendType::Faiss
                    ? vec_make_faiss_index(ctx->params)
                    : vec_make_hnswlib_index(ctx->params);
  if (!target) {
    ctx->build_in_progress = false;
    ib::warn() << "VECINDEX: function::vec_bg_build_task() Failed to create target index.";
    return;
  }

  target->train(ids.size(), xb.data());
  target->add(ids.size(), xb.data(), ids.data());

  std::string path;
  if (!vec_resolve_index_path(seg->aux_table_name, &path)) {
    ctx->build_in_progress = false;
    ib::warn() << "VECINDEX: function::vec_bg_build_task() Failed to resolve index path.";
    return;
  }

  target->save(path);
  seg->index_file_name = path;
  seg->index = std::move(target);

  ctx->build_in_progress = false;
  ib::warn() << "VECINDEX: function::vec_bg_build_task() Finished building index segment.";
}

}  // namespace

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
    vec_bg_build_task(ctx, staging_seg);
  }

  clear_pending();
}
