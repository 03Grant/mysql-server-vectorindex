#include "vec_ingest.h"
#include "dict0dict.h"
#include "trx0trx.h"
#include "vec_index.h"
#include "vec_tasks.h"
#include <unordered_set>
#include <vector>
#include <shared_mutex>

// 事务提交钩子里调用：清理事务上下文并触发必要的 rotate 任务
dberr_t vec_on_trx_commit(trx_t* trx) {
  if (trx == nullptr) {
    ib::warn() << "VECINDEX: vec_on_trx_commit called with null trx_t";
    return DB_ERROR;
  }

  auto* tctx = vec_lookup_trx_ctx(trx);
  if (tctx == nullptr) {
    return DB_SUCCESS;
  }

  std::unordered_set<dict_index_t*> touched_indexes;
  {
    std::lock_guard<std::mutex> lk(tctx->mu);
    touched_indexes = tctx->touched_indexes;
  }

  std::vector<dict_index_t*> flushed_indexes;

  // 成功后清空缓冲
  vec_trx_ctx_clear(tctx);

  // If exceed size, then submit tasks.
  if (!touched_indexes.empty()) {
    for (auto* idx : touched_indexes) {
      if (idx == nullptr) {
        continue;
      }
      flushed_indexes.push_back(idx);
    }
  }

  for (dict_index_t* idx : flushed_indexes) {
    if (idx == nullptr) {
      continue;
    }
    vec_index_ctx_t* ctx = idx->vec_runtime;
    if (ctx == nullptr) {
      continue;
    }
    std::shared_lock<std::shared_mutex> ctx_lock(ctx->mu);
    vec_index_segment_t* seg = ctx->mutable_segment();
    if (seg == nullptr || seg->index == nullptr) {
      continue;
    }
    const uint64_t limit = ctx->params.size;
    if (limit == 0) {
      continue;
    }
    const size_t current = seg->index->ntotal();
    if (current >= limit) {
      if (!ctx->is_rotation_pending.exchange(true)) {
        VecTaskManager::instance().submit_task(idx);
      }
    }
  }


  return DB_SUCCESS;
}

// 回滚路径
void vec_on_trx_rollback(trx_t* trx) {
  auto* tctx = vec_lookup_trx_ctx(trx);
  if (tctx != nullptr) {
    vec_trx_ctx_clear(tctx);
  }
}
