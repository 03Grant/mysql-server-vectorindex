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
#include <shared_mutex>
#include <string>
#include <unordered_map>
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

constexpr size_t kVecRollbackHexPreviewBytes = 32;

const char* vec_index_name(const dict_index_t* index) {
  return (index != nullptr && index->name != nullptr) ? index->name : "(null)";
}

std::string vec_hex_preview(const std::string& bytes, size_t max_bytes) {
  static constexpr char kHex[] = "0123456789abcdef";
  const size_t limit = bytes.size() < max_bytes ? bytes.size() : max_bytes;
  std::string out;
  out.reserve(limit * 2 + (bytes.size() > limit ? 3 : 0));
  for (size_t i = 0; i < limit; ++i) {
    const unsigned char ch = static_cast<unsigned char>(bytes[i]);
    out.push_back(kHex[ch >> 4]);
    out.push_back(kHex[ch & 0x0f]);
  }
  if (bytes.size() > limit) {
    out.append("...");
  }
  return out;
}

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

  auto* ctx = index->vec_runtime;
  if (!ctx || !ctx->inited) {
    ib::warn() << "VECINDEX: runtime context missing for index "
               << (index->name ? index->name : "(null)");
    return DB_ERROR;
  }

  std::shared_lock<std::shared_mutex> ctx_lock(ctx->mu);
  vec_index_segment_t* seg = ctx->mutable_segment();
  if (seg == nullptr || !seg->index) {
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
    std::unique_lock<std::shared_mutex> seg_lock(*seg->rw_lock);
    start = static_cast<int64_t>(flat->ntotal()); // 现有向量数
    for (size_t i = 0; i < k; ++i) {
      allocated_ids[i] = start + static_cast<int64_t>(i);
    }
    flat->add(k, xb.data(), allocated_ids.data()); // 内部分配 id: start..start+k-1
    // Move cache here 1/20/2026
    // To reduce times of acquire locks.

    for(size_t i = 0; i < k; ++i){
      const uint64_t faiss_id = static_cast<uint64_t>(allocated_ids[i]);
      const vec_item_t& it = bucket.items[i];
      // TODO write cache first, reduce lock time.
      dberr_t cache_err = DB_SUCCESS;
      
        //std::unique_lock<std::shared_mutex> seg_lock(*seg->rw_lock);
      cache_err = vec_insert_aux_cache(&seg->vid_pk_mapping, clust_index,
                                      faiss_id, it.pk_columns);
      seg->vecindex_bitmap.ensure_size(seg->vid_pk_mapping.size());
      
      if (cache_err != DB_SUCCESS) {
        ib::warn() << "VECINDEX: failed to insert aux cache entry for index "
                  << (index->name ? index->name : "(null)")
                  << " faiss_id=" << faiss_id;
        return cache_err;
      }
    }
    seg->vid_pk_mapping.ready = true;
  }
  // ib::warn() << "Step 2 vec_apply_bucket. ";

  // 3) 把主键快照与 start+i 写入辅助表
  // 1/20/2026 改在锁内写辅助表
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

  std::unordered_map<dict_index_t*, vec_trx_bucket_t> buckets;
  {
    std::lock_guard<std::mutex> lk(tctx->mu);
    buckets = tctx->by_index;
  }

  std::vector<dict_index_t*> flushed_indexes;
  for (auto& kv : buckets) {
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
  ib::warn() << "TEST_VEC_CONCUR: Trx_commit called!";

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
  auto* tctx = vec_lookup_trx_ctx(trx);
  ib::warn() << "VECINDEX_Rollback: vec_on_trx_rollback called for trx " << trx->id;
  if (tctx == nullptr) {
    ib::warn() << "VECINDEX_Rollback: no vec_trx_ctx_t for trx " << trx->id;
    return;
  }

  std::vector<vec_bitmap_undo_entry_t> undo_entries;
  std::vector<vec_update_undo_entry_t> update_entries;
  std::unordered_map<dict_index_t*, std::vector<std::string>> insert_keys;
  std::unordered_map<dict_index_t*, std::vector<std::string>> delete_keys;
  {
    std::lock_guard<std::mutex> lk(tctx->mu);
    undo_entries = tctx->bitmap_changes;
    update_entries = tctx->update_changes;
    for (const auto& kv : tctx->by_index) {
      if (kv.second.items.empty()) {
        continue;
      }
      auto& keys = insert_keys[kv.first];
      keys.reserve(kv.second.items.size());
      for (const auto& item : kv.second.items) {
        keys.push_back(item.pk_key);
      }
    }
    for (const auto& kv : tctx->deleted_pks_in_trx) {
      if (kv.second.empty()) {
        continue;
      }
      auto& keys = delete_keys[kv.first];
      keys.reserve(kv.second.size());
      for (const auto& key : kv.second) {
        keys.push_back(key);
      }
    }
  }

  for (const auto& kv : insert_keys) {
    const auto* index = kv.first;
    const auto& keys = kv.second;
    if (keys.empty()) {
      continue;
    }
    ib::warn() << "VEC_ROLLBACK_INSERT: index=" << vec_index_name(index)
               << " count=" << keys.size();
    for (const auto& key : keys) {
      ib::warn() << "VEC_ROLLBACK_INSERT: index=" << vec_index_name(index)
                 << " pk_key_bytes=" << key.size()
                 << " pk_key_hex=" << vec_hex_preview(key, kVecRollbackHexPreviewBytes);
    }
  }

  for (const auto& kv : delete_keys) {
    const auto* index = kv.first;
    const auto& keys = kv.second;
    if (keys.empty()) {
      continue;
    }
    ib::warn() << "VEC_ROLLBACK_DELETE: index=" << vec_index_name(index)
               << " count=" << keys.size();
    for (const auto& key : keys) {
      ib::warn() << "VEC_ROLLBACK_DELETE: index=" << vec_index_name(index)
                 << " pk_key_bytes=" << key.size()
                 << " pk_key_hex=" << vec_hex_preview(key, kVecRollbackHexPreviewBytes);
    }
  }

  for (const auto& entry : update_entries) {
    ib::warn() << "VEC_ROLLBACK_UPDATE: index=" << vec_index_name(entry.index)
               << " old_pk_bytes=" << entry.old_pk_key.size()
               << " old_pk_hex="
               << vec_hex_preview(entry.old_pk_key, kVecRollbackHexPreviewBytes)
               << " new_pk_bytes=" << entry.new_pk_key.size()
               << " new_pk_hex="
               << vec_hex_preview(entry.new_pk_key, kVecRollbackHexPreviewBytes)
               << " in_bucket=" << (entry.in_bucket ? 1 : 0);
  }

  for (auto it = undo_entries.rbegin(); it != undo_entries.rend(); ++it) {
    vec_index_ctx_t* vec_ctx = it->ctx;
    if (vec_ctx == nullptr) {
      continue;
    }

    std::lock_guard<std::shared_mutex> idx_lock(vec_ctx->mu);
    vec_index_segment_t* seg = nullptr;
    for (auto& s : vec_ctx->segments) {
      if (s.vecindex_id == it->segment_id) {
        seg = &s;
        break;
      }
    }
    if (seg == nullptr) {
      seg = vec_ctx->mutable_segment();
      if (seg != nullptr && seg->vecindex_id != it->segment_id &&
          it->segment_id != 0) {
        seg = nullptr;
      }
    }

    if (seg != nullptr) {
      ib::warn() << "VEC_ROLLBACK_DELETE: bitmap restore index="
                 << vec_index_name(it->index)
                 << " seg_id=" << it->segment_id
                 << " vid=" << it->vid
                 << " old_val=" << it->old_val;
      seg->vecindex_bitmap.set(static_cast<size_t>(it->vid), it->old_val);
    }else{
      // TODO deal with the case where the segment is not found
      
      ib::warn() << "VEC_ROLLBACK_DELETE: bitmap restore skipped index="
                 << vec_index_name(it->index)
                 << " seg_id=" << it->segment_id
                 << " vid=" << it->vid
                 << " old_val=" << it->old_val;
    }

  }

  vec_trx_ctx_clear(tctx); // 没有改 Faiss/表，自然丢弃
}
