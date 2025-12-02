#include "vec_ingest.h"
#include "vec_id_alloc.h"          // 你的分配器
#include "vec_aux_tables.h"        // 你的辅助表写入 API
#include "dict0dict.h"
#include "trx0trx.h"
#include "trx0roll.h"
#include "current_thd.h"
#include "ha_innodb.h"            // thd_to_trx
#include "ut0dbg.h"
#include "vec_index.h"
#include "vec_tasks.h"
#include <omp.h>

#include <cstring>
#include <vector>

namespace {

// Simple RAII guard to set per-operation thread count for vector libraries.
// Currently both Faiss and HNSWLIB rely on OpenMP for add/training.
class ScopedVecThreads {
 public:
  explicit ScopedVecThreads(const vec_params_t* params) {
#ifdef _OPENMP
    if (params != nullptr && params->build_threads > 0) {
      prev_threads_ = omp_get_max_threads();
      omp_set_num_threads(params->build_threads);
      changed_ = true;
    }
#else
    (void)params;
#endif
  }

  ~ScopedVecThreads() {
#ifdef _OPENMP
    if (changed_) {
      omp_set_num_threads(prev_threads_);
    }
#endif
  }

 private:
#ifdef _OPENMP
  int  prev_threads_{1};
  bool changed_{false};
#endif
};

} // namespace

// 批量add_with_ids，再insert辅助表
static dberr_t vec_apply_bucket(trx_t* exec_trx, vec_trx_bucket_t& bucket) {
  dict_index_t* index = bucket.index;
  if (!index || bucket.items.empty()) {
    return DB_SUCCESS;
  }

  if (exec_trx == nullptr) {
    ib::warn() << "VECINDEX: exec_trx is null for index "
               << (index->name ? index->name : "(null)");
    return DB_ERROR;
  }


  // ib::warn() << "Step 1 vec_apply_bucket. ";


  auto* ctx = index->vec_runtime;
  vec_index_segment_t* seg = ctx ? ctx->mutable_segment() : nullptr;
  if (!ctx || !ctx->inited || seg == nullptr || !seg->index) {
    ib::warn() << "VECINDEX: runtime context missing for index " << (index->name ? index->name : "(null)");
    return DB_ERROR;
  }

  dict_table_t* base_table = index->table;
  dict_index_t* clust_index = base_table ? base_table->first_index() : nullptr;
  if (clust_index == nullptr) {
    ib::warn() << "VECINDEX: clustered index missing for base table of "
               << (index->name ? index->name : "(null)");
    return DB_ERROR;
  }

  const size_t dim = size_t(bucket.dim);
  const size_t k   = bucket.items.size();

  // 1) 打平成连续 xb
  std::vector<float> xb(k * dim);
  for (size_t i = 0; i < k; ++i) {
    std::memcpy(xb.data() + i * dim, bucket.items[i].vec.data(), dim * sizeof(float));
  }

  // 2) 粗锁下：读取起始 id = ntotal，执行 add(k, xb)
  IVectorIndex* flat = seg->index.get();
  int64_t start = 0;
  std::vector<int64_t> allocated_ids(k);
  {
    ScopedVecThreads guard(index->vec_params);
    std::lock_guard<std::mutex> g(ctx->mu);
    start = static_cast<int64_t>(flat->ntotal()); // 现有向量数
    for (size_t i = 0; i < k; ++i) {
      allocated_ids[i] = start + static_cast<int64_t>(i);
    }
    flat->add(k, xb.data(), allocated_ids.data()); // 内部分配 id: start..start+k-1
  }
  // ib::warn() << "Step 2 vec_apply_bucket. ";

  // 3) 锁外写辅助表：把主键快照与 start+i 写入辅助表（与事务同生死）

  vec_aux_mode_t aux_mode = bucket.aux_mode;
  if (aux_mode == vec_aux_mode_t::UNKNOWN) {
    aux_mode = vec_aux_mode_t::DIRECT_INSERT;
  }

  for (size_t i = 0; i < k; ++i) {
    const uint64_t faiss_id = static_cast<uint64_t>(allocated_ids[i]);
    const vec_item_t& it = bucket.items[i];

    dberr_t last_err = DB_SUCCESS;
    if (aux_mode == vec_aux_mode_t::PREINSERT_NULL) {
      last_err = vec_aux_update_pk_vid(exec_trx, index, it.pk_columns, faiss_id);
    } else {
      last_err = vec_aux_insert_one(exec_trx, index, it.pk_columns, faiss_id);
    }
    if (last_err != DB_SUCCESS) {
      ib::warn() << "VECINDEX: aux insert failed for index "
                 << (index->name ? index->name : "(null)")
                 << " faiss_id=" << faiss_id
                 << " error=" << last_err;
      return last_err;
    }

    dberr_t cache_err =
        vec_insert_aux_cache(&seg->aux_cache, clust_index, faiss_id,
                             it.pk_columns);
    if (cache_err != DB_SUCCESS) {
      ib::warn() << "VECINDEX: failed to insert aux cache entry for index "
                 << (index->name ? index->name : "(null)")
                 << " faiss_id=" << faiss_id;
      return cache_err;
    }
  }

  return DB_SUCCESS;
}

// 事务提交钩子里调用：遍历所有桶应用
dberr_t vec_on_trx_commit(trx_t* trx) {
  auto* tctx = vec_get_or_create_trx_ctx(trx);
  if (!tctx) {
    ib::warn() << "VECINDEX: cannot get vec_trx_ctx_t";
    return DB_ERROR;
  } 
  if (trx == nullptr) {
    ib::warn() << "VECINDEX: vec_on_trx_commit called with null trx_t";
    return DB_ERROR;
  }

  trx_t* exec_trx = trx;
  trx_t* background = nullptr;
  bool owns_exec_trx = false;
  dberr_t last_err = DB_SUCCESS;

  if (trx->state.load(std::memory_order_acquire) != TRX_STATE_ACTIVE) {
    background = trx_allocate_for_background();
    if (background == nullptr) {
      ib::warn() << "VECINDEX: failed to allocate background transaction for aux insert";
      return DB_ERROR;
    }
    trx_start_internal(background, UT_LOCATION_HERE);
    exec_trx = background;
    owns_exec_trx = true;
  }

  ib::warn() << "VECINDEX: flushing vector rows for trx " << trx->id
             << " using exec_trx " << exec_trx->id;

  std::vector<dict_index_t*> flushed_indexes;
  for (auto& kv : tctx->by_index) {
    last_err = vec_apply_bucket(exec_trx, kv.second);
    if (last_err != DB_SUCCESS) {
      ib::warn() << "VECINDEX: vec_apply_bucket failed with error " << last_err;
      vec_trx_ctx_clear(tctx);
      break;
    }
    flushed_indexes.push_back(kv.first);
  }


  DEBUG_SYNC_C("vec_aux_before_aux_commit");
  DBUG_EXECUTE_IF("crash_vec_aux_before_aux_commit", DBUG_SUICIDE(););


  if (owns_exec_trx) {
    if (last_err == DB_SUCCESS) {
      dberr_t commit_err = trx_commit_for_mysql(exec_trx);
      if (commit_err != DB_SUCCESS) {
        ib::warn() << "VECINDEX: commit of background aux inserts failed with error "
                   << commit_err;
        last_err = commit_err;
      }
    } else {
      /* Background trx is not on mysql_trx_list; use savepoint rollback to
      avoid the in_mysql_trx_list assertion. */
      trx_rollback_to_savepoint(exec_trx, nullptr);
    }
    trx_free_for_background(exec_trx);
  }

  if (last_err != DB_SUCCESS) {
    return last_err;
  }

  DEBUG_SYNC_C("vec_aux_after_aux_commit");
  DBUG_EXECUTE_IF("crash_vec_aux_after_aux_commit", DBUG_SUICIDE(););

  // 成功后清空缓冲
  vec_trx_ctx_clear(tctx);

  // If exceed size, then submit tasks.
  for (dict_index_t* idx : flushed_indexes) {
    if (idx == nullptr) {
      continue;
    }
    vec_index_ctx_t* ctx = idx->vec_runtime;
    vec_index_segment_t* seg = ctx ? ctx->mutable_segment() : nullptr;
    if (ctx == nullptr || seg == nullptr || seg->index == nullptr) {
      continue;
    }
    const uint64_t limit = ctx->params.size;
    if (limit == 0) {
      continue;
    }
    const size_t current = seg->index->ntotal();
    if (current >= limit) {
      if (!vec_prepare_pending_mem_table(idx)) {
        ib::warn() << "VECINDEX: failed to prepare pending aux table for index "
                   << (idx->name ? idx->name : "(null)");
        continue;
      }
      if (!ctx->is_rotation_pending.exchange(true)) {
        VecTaskManager::instance().submit_task(idx);
      }
    }
  }


  return DB_SUCCESS;
}

// 回滚路径
void vec_on_trx_rollback(trx_t* trx) {
  auto* tctx = vec_get_or_create_trx_ctx(trx);
  if (!tctx) return;
  vec_trx_ctx_clear(tctx); // 没有改 Faiss/表，自然丢弃
}
