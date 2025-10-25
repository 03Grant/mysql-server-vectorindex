#include "vec_ingest.h"
#include "vec_id_alloc.h"          // 你的分配器
#include "vec_aux_tables.h"        // 你的辅助表写入 API
#include "dict0dict.h"
#include "trx0trx.h"
#include "trx0roll.h"
#include "ut0dbg.h"
#include <faiss/Index.h>

#include <cstring>

// 批量落一“桶”：先 add_with_ids 到 m 段，再写辅助表
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


  ib::warn() << "Step 1 vec_apply_bucket. ";


  auto* ctx = index->vec_runtime;
  if (!ctx || !ctx->inited || ctx->indices.empty() || !ctx->indices[0]) {
    ib::warn() << "VECINDEX: runtime context missing for index " << (index->name ? index->name : "(null)");
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
  faiss::Index* flat = ctx->indices[0].get();
  faiss::idx_t start = 0;
  {
    std::lock_guard<std::mutex> g(ctx->mu);
    start = static_cast<faiss::idx_t>(flat->ntotal); // 现有向量数
    flat->add(k, xb.data());                         // 内部分配 id: start..start+k-1
  }
  ib::warn() << "Step 2 vec_apply_bucket. ";

  // 3) 锁外写辅助表：把主键快照与 start+i 写入辅助表（与事务同生死）

  
  for (size_t i = 0; i < k; ++i) {
    const uint64_t faiss_id = static_cast<uint64_t>(start + static_cast<faiss::idx_t>(i));
    const vec_item_t& it = bucket.items[i];

    dberr_t last_err = vec_aux_insert_one(exec_trx, index, it.pk_columns, faiss_id);
    if (last_err != DB_SUCCESS) {
      ib::warn() << "VECINDEX: aux insert failed for index "
                 << (index->name ? index->name : "(null)")
                 << " faiss_id=" << faiss_id
                 << " error=" << last_err;
      return last_err;
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
    exec_trx = background;
    owns_exec_trx = true;
  }

  for (auto& kv : tctx->by_index) {
    last_err = vec_apply_bucket(exec_trx, kv.second);
    if (last_err != DB_SUCCESS) {
      ib::warn() << "VECINDEX: vec_apply_bucket failed with error " << last_err;
      break;
    }
  }
  // 
  if (owns_exec_trx) {
    if (last_err == DB_SUCCESS) {
      dberr_t commit_err = trx_commit_for_mysql(exec_trx);
      if (commit_err != DB_SUCCESS) {
        ib::warn() << "VECINDEX: commit of background aux inserts failed with error "
                   << commit_err;
        last_err = commit_err;
      }
    } else {
      trx_rollback_for_mysql(exec_trx);
    }
    trx_free_for_background(exec_trx);
  }

  if (last_err != DB_SUCCESS) {
    return last_err;
  }

  // 成功后清空缓冲
  vec_trx_ctx_clear(tctx);
  return DB_SUCCESS;
}

// 回滚路径
void vec_on_trx_rollback(trx_t* trx) {
  auto* tctx = vec_get_or_create_trx_ctx(trx);
  if (!tctx) return;
  vec_trx_ctx_clear(tctx); // 没有改 Faiss/表，自然丢弃
}
