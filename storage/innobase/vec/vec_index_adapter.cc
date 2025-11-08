#pragma once
// vec_index_adapter.cc
#include "vec_index_adapter.h"
#include "vec_index_runtime.h"
#include "vec_faiss_factory.h"
#include "vec_faiss_includes.h"

#include "dict0mem.h"
#include "dict0dd.h"
#include "ut0ut.h"
#include "my_sys.h"
#include "mysqld.h"         // reg_ext_length
#include "sql/handler.h"
#include "sql/sql_class.h"     // THD, current_thd
#include "sql/sql_base.h"      // open_table_uncached, intern_close_table
#include "sql/sql_error.h"     // push_warning_printf
#include "sql/sql_table.h"     // build_table_filename
#include "sql/table.h"
#include "sql/mdl.h"
#include "mysqld_error.h"      // ER_UNKNOWN_ERROR

#include <cstring>
#include <string>
#include <algorithm>
#include <limits>

struct dict_index_t;

namespace {

struct AuxNameParts {
  std::string db;
  std::string table;
};

std::string vec_aux_table_name(const dict_index_t *index) {
  if (index == nullptr || index->table == nullptr ||
      index->table->name.m_name == nullptr) {
    return {};
  }

  const char *parent = index->table->name.m_name;
  const char *slash = std::strchr(parent, '/');
  if (slash == nullptr || slash == parent) {
    return {};
  }

  std::string db(parent, static_cast<size_t>(slash - parent));
  if (db.empty()) return {};

  std::string suffix("I_VEC_");
  suffix.append(std::to_string(
      static_cast<unsigned long long>(index->table->id)));
  suffix.push_back('_');
  suffix.append(std::to_string(
      static_cast<unsigned long long>(index->id)));

  std::string full;
  full.reserve(db.size() + 1 + suffix.size());
  full.append(db);
  full.push_back('/');
  full.append(suffix);
  return full;
}

bool vec_parse_aux_table_name(const std::string &full, AuxNameParts *parts) {
  if (parts == nullptr) return false;
  const auto slash = full.find('/');
  if (slash == std::string::npos || slash == 0 || slash + 1 >= full.size()) {
    return false;
  }

  parts->db = full.substr(0, slash);
  parts->table = full.substr(slash + 1);
  return !(parts->db.empty() || parts->table.empty());
}

}  // namespace

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
  // vec_params_t im_mem_p = {type_tag: VEC_T_FLAT,
  //                          metric_tag: p.metric_tag,
  //                          dim: p.dim,
  //                          size: 0,  // in-mem index has no size limit
  //                          build_threads: p.build_threads,
  //                          nlist: 0,
  //                          m: 0,
  //                          nbits: 0,
  //                          hnsw_m: 0,
  //                          efConstruction: 0};

  vec_params_t im_mem_p = {type_tag: VEC_T_HNSW,
                           metric_tag: p.metric_tag,
                           dim: p.dim,
                           size: 0,  // in-mem index has no size limit
                           build_threads: p.build_threads,
                           nlist: 0,
                           m: 0,
                           nbits: 0,
                           hnsw_m: 32,
                           efConstruction: 128};

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

dberr_t vec_open_aux_table(dict_index_t *idx) {
  if (idx == nullptr || idx->vec_runtime == nullptr) {
    return DB_ERROR;
  }

  vec_index_ctx_t *ctx = idx->vec_runtime;
  if (ctx->aux_dict_table != nullptr) {
    return DB_SUCCESS;
  }

  const std::string aux_name = vec_aux_table_name(idx);
  if (aux_name.empty()) {
    ib::warn() << "VECINDEX: failed to derive auxiliary table name for index "
               << (idx->name ? idx->name : "(null)");
    return DB_FAIL;
  }

  if (ctx->aux_dict_table == nullptr) {
    dict_table_t *aux_table =
        dd_table_open_on_name_in_mem(aux_name.c_str(), false);
    if (aux_table == nullptr) {
      ib::warn() << "VECINDEX: auxiliary table '" << aux_name
                 << "' is not available in dictionary cache";
      return DB_FAIL;
    }
    ctx->aux_dict_table = aux_table;
  }

  return DB_SUCCESS;
}

dberr_t vec_open_aux_table_for_thd(dict_index_t *idx, THD *thd,
                                   vec_aux_table_handle *handle) {
  if (handle == nullptr) return DB_ERROR;
  handle->table = nullptr;
  handle->se_handler = nullptr;
  handle->mdl_ticket = nullptr;

  if (idx == nullptr || idx->vec_runtime == nullptr || thd == nullptr) {
    return DB_ERROR;
  }

  vec_index_ctx_t *ctx = idx->vec_runtime;
  dberr_t dict_err = vec_open_aux_table(idx);
  if (dict_err != DB_SUCCESS) {
    ib::warn() << "VECINDEX: unable to cache auxiliary dictionary table for index '"
               << (idx->name ? idx->name : "(null)")
               << "', err=" << dict_err << ". Continuing with per-THD open.";
  } else {
    dict_table_t *dict_table = ctx->aux_dict_table;
    // ib::warn() << "VECINDEX: cached auxiliary dict table for index '"
    //            << (idx->name ? idx->name : "(null)")
    //            << "' ref_count="
    //            << (dict_table != nullptr ? dict_table->get_ref_count() : 0)
    //            << " ctx=" << ctx;
  }

  const std::string aux_name = vec_aux_table_name(idx);

  AuxNameParts parts;
  if (!vec_parse_aux_table_name(aux_name, &parts)) {
    ib::warn() << "VECINDEX: malformed auxiliary table name '" << aux_name
               << "'";
    return DB_FAIL;
  }

  MDL_request mdl_request;
  MDL_REQUEST_INIT(&mdl_request, MDL_key::TABLE, parts.db.c_str(),
                   parts.table.c_str(), MDL_SHARED_READ, MDL_TRANSACTION);

  if (thd->mdl_context.acquire_lock(&mdl_request,
                                    thd->variables.lock_wait_timeout)) {
    ib::warn() << "VECINDEX: failed to acquire MDL for auxiliary table '"
               << aux_name << "'";
    return DB_LOCK_WAIT_TIMEOUT;
  }

  auto release_mdl = [&]() {
    if (mdl_request.ticket != nullptr) {
      if (MDL_context *mdl_ctx = mdl_request.ticket->get_ctx()) {
        mdl_ctx->release_lock(mdl_request.ticket);
      }
      mdl_request.ticket = nullptr;
    }
  };

  dd::cache::Dictionary_client::Auto_releaser releaser(thd->dd_client());
  const dd::Table *dd_table_obj = nullptr;
  if (thd->dd_client()->acquire(parts.db.c_str(), parts.table.c_str(),
                                &dd_table_obj)) {
    ib::warn() << "VECINDEX: failed to acquire DD object for auxiliary table '"
               << aux_name << "'";
    release_mdl();
    return DB_ERROR;
  }

  char path[FN_REFLEN + 1]{};
  bool truncated = false;
  size_t path_len =
      build_table_filename(path, sizeof(path) - 1 - reg_ext_length,
                           parts.db.c_str(), parts.table.c_str(), "", 0,
                           &truncated);
  if (path_len == 0 || truncated) {
    ib::warn() << "VECINDEX: failed to build path for auxiliary table '"
               << aux_name << "'";
    release_mdl();
    return DB_FAIL;
  }

  TABLE *mysql_table =
      open_table_uncached(thd, path, parts.db.c_str(), parts.table.c_str(),
                          false, true, *dd_table_obj);
  if (mysql_table == nullptr) {
    ib::warn() << "VECINDEX: open_table_uncached() failed for auxiliary table '"
               << aux_name << "'";
    release_mdl();
    return DB_ERROR;
  }

  handler *opened_handler = mysql_table->file;
  if (opened_handler == nullptr) {
    ib::warn() << "VECINDEX: auxiliary table handler missing for '" << aux_name
               << "'";
    intern_close_table(mysql_table);
    release_mdl();
    return DB_ERROR;
  }

  handle->table = mysql_table;
  handle->se_handler = opened_handler;
  handle->mdl_ticket = mdl_request.ticket;
  mdl_request.ticket = nullptr;

  dict_table_t *dict_table = ctx->aux_dict_table;
  // ib::warn() << "VECINDEX: open_aux_table_for_thd success idx='"
  //            << (idx->name ? idx->name : "(null)")
  //            << "' aux='" << aux_name << "' thd=" << thd
  //            << " table=" << mysql_table << " handler=" << opened_handler
  //            << " dict_ref=" << (dict_table != nullptr
  //                                    ? dict_table->get_ref_count()
  //                                    : 0);

  return DB_SUCCESS;
}

void vec_close_aux_table_for_thd(THD *thd, vec_aux_table_handle *handle) {
  if (handle == nullptr) return;

  // ib::warn() << "VECINDEX: close_aux_table_for_thd thd=" << thd
  //            << " table=" << handle->table
  //            << " mdl=" << handle->mdl_ticket;

  // if (handle->table != nullptr) {
  //   intern_close_table(handle->table);
  //   handle->table = nullptr;
  //   handle->se_handler = nullptr;
  // }

  if (handle->mdl_ticket != nullptr) {
    if (thd != nullptr) {
      thd->mdl_context.release_lock(handle->mdl_ticket);
    } else {
      if (MDL_context *mdl_ctx = handle->mdl_ticket->get_ctx()) {
        mdl_ctx->release_lock(handle->mdl_ticket);
      }
    }
    handle->mdl_ticket = nullptr;
  }

  // ib::warn() << "VECINDEX: close_aux_table_for_thd completed.";
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

void vec_destroy(vec_index_ctx_t* ctx) {
  if (ctx == nullptr) return;
  if (ctx->aux_dict_table != nullptr) {
    dd_table_close(ctx->aux_dict_table, nullptr, nullptr, false);
    ctx->aux_dict_table = nullptr;
  }
  delete ctx;
}

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
