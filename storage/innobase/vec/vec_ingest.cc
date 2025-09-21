#include "vec_txn_buf.h"
#include "vec_id_alloc.h"          // 你的分配器
#include "vec_aux_tables.h"        // 你的辅助表写入 API
#include "dict0dict.h"
#include "ut0dbg.h"
#include <faiss/Index.h>

#include <cstring>

// 批量落一“桶”：先 add_with_ids 到 m 段，再写辅助表
static int vec_apply_bucket(trx_t* trx, vec_trx_bucket_t& bucket) {
  dict_index_t* index = bucket.index;
  if (!index || bucket.items.empty()) return 0;

  // 取运行时上下文
  auto* ctx = index->vec_runtime;             // 你在 create 时挂上的
  if (!ctx || !ctx->inited || ctx->indices.empty() || !ctx->indices[0]) {
    return -10;
  }

  const size_t dim = size_t(bucket.dim);
  const size_t k   = bucket.items.size();

  // 1) 构造连续的 xb 与 ids（注意 idx_t 有符号 64 位）
  std::vector<float> xb; xb.resize(k * dim);
  std::vector<faiss::idx_t> ids; ids.resize(k);
  for (size_t i = 0; i < k; ++i) {
    std::memcpy(xb.data() + i * dim, bucket.items[i].vec.data(), dim * sizeof(float));
  }

  // 2) 懒恢复 + 批量预留 ID（短时间持锁）
  {
    std::lock_guard<std::mutex> g(ctx->mu);
    // 懒恢复（从 aux 读 MAX，空表则从 0 开始）
    dberr_t e = ctx->id_alloc.recover_from_aux(trx, index);
    if (e != DB_SUCCESS) return -11;

    uint64_t start = 0;
    e = ctx->id_alloc.reserve(uint64_t(k), &start);
    if (e != DB_SUCCESS) return -12;

    for (size_t i = 0; i < k; ++i) ids[i] = static_cast<faiss::idx_t>(start + i);

    // 3) 粗锁下直接 add 到内存段（indices[0]）
    ctx->indices[0]->add_with_ids(k, xb.data(), ids.data());
  }

  // 4) 写入辅助表（同一事务内；失败则回滚整事务，允许 ID 空洞）
  //    批量插入 (pk_bin, faiss_id)
  //    你可以一条条 insert 或 row API 一次性写多行
  for (size_t i = 0; i < k; ++i) {
    dberr_t e = vec_aux_insert_one(trx, index,
                                   bucket.items[i].pk_bin.data(),
                                   bucket.items[i].pk_bin.size(),
                                   static_cast<uint64_t>(ids[i]));
    if (e != DB_SUCCESS) {
      return -13;
    }
  }

  return 0;
}

// 事务提交钩子里调用：遍历所有桶应用
int vec_on_trx_commit(trx_t* trx) {
  auto* tctx = vec_get_or_create_trx_ctx(trx);
  if (!tctx) return 0;

  for (auto& kv : tctx->by_index) {
    int rc = vec_apply_bucket(trx, kv.second);
    if (rc != 0) {
      // 把错误转 dberr_t 或日志；返回非 0 让上层回滚
      return rc;
    }
  }

  // 成功后清空缓冲
  vec_trx_ctx_clear(tctx);
  return 0;
}

// 回滚路径
void vec_on_trx_rollback(trx_t* trx) {
  auto* tctx = vec_get_or_create_trx_ctx(trx);
  if (!tctx) return;
  vec_trx_ctx_clear(tctx); // 没有改 Faiss/表，自然丢弃
}
