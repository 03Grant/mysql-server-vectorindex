#pragma once
// vec_index_adapter.cc
#include "vec_index_adapter.h"
#include "vec_index_runtime.h"
#include "vec_faiss_factory.h"
#include "vec_faiss_includes.h"

#include "dict0mem.h" 
#include "ut0ut.h"
#include "sql/sql_class.h"     // THD, current_thd
#include "sql/sql_error.h"     // push_warning_printf
#include "mysqld_error.h"      // ER_UNKNOWN_ERROR



struct dict_index_t; 

dberr_t vec_create_index_low(dict_index_t* idx) {
  if (idx == nullptr) return DB_FAIL;

  // 1 ) get params
  vec_params_t p{};
  if (idx->vec_params) {
    p = *idx->vec_params;
  } else {
    // if no params, fail
    ib::warn() << "VECINDEX checkpoint: no vec_params found, fail. "
               << "table=" << (idx->table ? idx->table->name.m_name : "(null)")
               << ", index_name=" << (idx->name ? idx->name : "(null)")
               << ", index_id=" << (unsigned long long)idx->id;
    return DB_FAIL;
  }

  // 2) create default runtime context 
  vec_params_t im_mem_p = {type_tag: VEC_T_FLAT,
                           metric_tag: p.metric_tag,
                           dim: p.dim,
                           size: 0,  // in-mem index has no size limit
                           build_threads: p.build_threads,
                           nlist: 0,
                           m: 0,
                           nbits: 0,
                           hnsw_m: 0,
                           efConstruction: 0};
  std::unique_ptr<vec_index_ctx_t> ctx = vec_create(im_mem_p);
  if (!ctx || !ctx->inited) {
    ib::warn() << "VECINDEX checkpoint: vec_create() failed. "
               << "table=" << (idx->table ? idx->table->name.m_name : "(null)")
               << ", index_name=" << (idx->name ? idx->name : "(null)")
               << ", index_id=" << (unsigned long long)idx->id;
    return DB_FAIL;
  }

  idx->vec_runtime = ctx.release();

    // 3) TODO: there will be more than one index pointers in the context. Handle it!

  return DB_SUCCESS;
}

std::unique_ptr<vec_index_ctx_t> vec_create(const vec_params_t& p) {
  auto ctx = std::make_unique<vec_index_ctx_t>();
  ctx->params = p;

  auto index = vec_make_faiss_index(p);

  if (!index) {
    ib::warn() << "VECINDEX: vec_make_faiss_index() failed.";
    ctx->inited = false;
    return ctx;
  }

  ctx->indices.push_back(std::move(index));
  ctx->inited = true;
  return ctx;
}

bool vec_drop_index(vec_index_ctx_t& ctx, size_t seg_idx, bool allow_drop_mutable){
  std::lock_guard<std::mutex> lk(ctx.mu);
  if (seg_idx >= ctx.indices.size()) return false;
  if (seg_idx == 0 && !allow_drop_mutable) return false;

  // erase to release the memory
  ctx.indices.erase(ctx.indices.begin() + seg_idx);
  return true;

}

void vec_destroy(vec_index_ctx_t* ctx) { delete ctx; }

int vec_add(vec_index_ctx_t& ctx, const float* xb, size_t n) {
  std::lock_guard<std::mutex> lk(ctx.mu);
  if (!ctx.inited || ctx.indices.empty() || !ctx.indices[0]) return -1;
  // 假设 dim 匹配，由你在外面保证；FAISS 可能会抛异常，后续你可以做 try/catch
  ctx.indices[0]->add(n, xb);
  return n;
}

int vec_add_with_ids(vec_index_ctx_t& ctx, const float* xb, const faiss::idx_t* ids, size_t n){
  std::lock_guard<std::mutex> lk(ctx.mu);
  if (!ctx.inited || ctx.indices.empty() || !ctx.indices[0]) return -1;

  ctx.indices[0]->add_with_ids(n, xb, ids);
  return 0;
}


static inline bool is_min_better(const vec_params_t& params) {
  return !(params.metric_tag == VEC_M_IP || params.metric_tag == VEC_M_COSINE);
}

int vec_search(vec_index_ctx_t& ctx,
               const float* q, size_t nq, size_t k,
               float* D_out, faiss::idx_t* I_out)
{
  std::lock_guard<std::mutex> lk(ctx.mu);
  if (!ctx.inited || ctx.indices.empty()) return -1;

  const bool prefer_small = is_min_better(ctx.params);
  const size_t nseg = ctx.indices.size();

  // MVP：按“每个查询”循环，便于把各段结果做 k-way 合并
  for (size_t qi = 0; qi < nq; ++qi) {
    // 收集所有段的候选
    std::vector<std::pair<float, faiss::idx_t>> cand;
    cand.reserve(nseg * k);

    const float* qvec = q + qi * ctx.params.dim;

    for (size_t s = 0; s < nseg; ++s) {
      auto* seg = ctx.indices[s].get();
      if (!seg) continue;

      std::vector<float>  D(k);
      std::vector<faiss::idx_t> I(k);

      seg->search(1, qvec, k, D.data(), I.data());

      // 过滤掉无效 id（Faiss 可能返回 -1 表示候选不足）
      for (size_t t = 0; t < k; ++t) {
        if (I[t] >= 0) cand.emplace_back(D[t], I[t]);
      }
    }

    // cand 可能 < k（所有段都少），也可能 > k（段数多）
    if (cand.empty()) {
      // 全部填充为 “空”
      std::fill_n(D_out + qi * k, k, prefer_small ? std::numeric_limits<float>::infinity()
                                                  : -std::numeric_limits<float>::infinity());
      std::fill_n(I_out + qi * k, k, faiss::idx_t(-1));
      continue;
    }

    // 取全局 top-K
    if (cand.size() > k) {
      if (prefer_small) {
        std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                          [](auto& a, auto& b){ return a.first < b.first; });
        cand.resize(k);
      } else {
        std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                          [](auto& a, auto& b){ return a.first > b.first; });
        cand.resize(k);
      }
    } else {
      if (prefer_small) {
        std::sort(cand.begin(), cand.end(),
                  [](auto& a, auto& b){ return a.first < b.first; });
      } else {
        std::sort(cand.begin(), cand.end(),
                  [](auto& a, auto& b){ return a.first > b.first; });
      }
      // if cand.size() < k，fill others with invalid values
    }

    // 写回输出
    float*       Dq = D_out + qi * k;
    faiss::idx_t* Iq = I_out + qi * k;

    size_t m = std::min(k, cand.size());
    for (size_t t = 0; t < m; ++t) {
      Dq[t] = cand[t].first;
      Iq[t] = cand[t].second;
    }
    // 不足部分用“空”填充
    for (size_t t = m; t < k; ++t) {
      Dq[t] = prefer_small ? std::numeric_limits<float>::infinity()
                           : -std::numeric_limits<float>::infinity();
      Iq[t] = faiss::idx_t(-1);
    }
  }

  return 0;
}
