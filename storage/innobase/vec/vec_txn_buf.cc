#include "vec_txn_buf.h"

#include "dict0dict.h"
#include "mem0mem.h"
#include "row0row.h"
#include "ut0dbg.h"

#include "data0data.h"
#include "my_byteorder.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

std::mutex g_trx_ctx_mu;
std::unordered_map<trx_t *, std::unique_ptr<vec_trx_ctx_t>> g_trx_ctx_map;

// —— 聚簇主键逐列拷贝 ——
static bool vec_capture_pk_columns(dict_table_t *table, const dtuple_t *row,
                                   std::vector<vec_pk_column_t> &out) {
  if (table == nullptr || row == nullptr) {
    return false;
  }

  dict_index_t *clust = table->first_index();   // get clustered index
  if (clust == nullptr) {
    return false;
  }

  mem_heap_t *heap = mem_heap_create(512, UT_LOCATION_HERE);
  if (heap == nullptr) {
    return false;
  }

  dtuple_t *entry = row_build_index_entry_low(row, nullptr, clust, heap,
                                              ROW_BUILD_FOR_INSERT);
  if (entry == nullptr) {
    mem_heap_free(heap);
    return false;
  }

  const ulint n_fields = clust->n_fields;
  out.clear();
  out.resize(n_fields);

  for (ulint i = 0; i < n_fields; ++i) {
    const dfield_t *df = dtuple_get_nth_field(entry, i);
    if (df == nullptr) {
      out.clear();
      mem_heap_free(heap);
      return false;
    }

    vec_pk_column_t &col = out[i];
    col.is_null = dfield_is_null(df);

    const dict_field_t *ind_field = clust->get_field(i);
    if (ind_field != nullptr && ind_field->col != nullptr) {
      col.mtype = ind_field->col->mtype;
      col.prtype = ind_field->col->prtype;
    } else {
      col.mtype = DATA_FIXBINARY;
      col.prtype = 0;
    }

    if (col.is_null) {
      col.data.clear();
      continue;
    }

    const unsigned char *data =
        static_cast<const unsigned char *>(dfield_get_data(df));
    const ulint len = dfield_get_len(df);
    if (len > 0 && data == nullptr) {
      out.clear();
      mem_heap_free(heap);
      return false;
    }

    col.data.assign(data, data + len);
  }

  mem_heap_free(heap);
  return true;
}


// —— 抽取向量字节并校验 ——
static bool vec_extract_and_validate(const dfield_t *field, unsigned dim,
                                     std::vector<float> &out) {
  if (field == nullptr || dim == 0 || dfield_is_null(field)) {
    return false;
  }

  const size_t expect_bytes = static_cast<size_t>(dim) * sizeof(float);
  const ulint raw_len = dfield_get_len(field);
  const unsigned char *raw =
      static_cast<const unsigned char *>(dfield_get_data(field));

  if (raw == nullptr || raw_len != expect_bytes) {
    return false;
  }

  out.resize(dim);
  for (unsigned i = 0; i < dim; ++i) {
    const float value = float4get(raw + i * sizeof(float));
    if (!std::isfinite(value)) {
      return false;
    }
    out[i] = value;
  }

  return true;
}

}  // namespace

// —— 事务 ctx 管理 ——
vec_trx_ctx_t *vec_get_or_create_trx_ctx(trx_t *trx) {
  if (trx == nullptr) {
    return nullptr;
  }

  std::lock_guard<std::mutex> guard(g_trx_ctx_mu);
  auto it = g_trx_ctx_map.find(trx);
  if (it != g_trx_ctx_map.end()) {
    return it->second.get();
  }

  auto ctx = std::make_unique<vec_trx_ctx_t>();
  ctx->owner = trx;
  auto *raw = ctx.get();
  g_trx_ctx_map.emplace(trx, std::move(ctx));
  return raw;
}

void vec_trx_ctx_clear(vec_trx_ctx_t *ctx) {
  if (ctx == nullptr) {
    return;
  }

  trx_t *owner = ctx->owner;
  if (owner == nullptr) {
    ctx->by_index.clear();
    return;
  }

  std::unique_ptr<vec_trx_ctx_t> owned;
  {
    std::lock_guard<std::mutex> guard(g_trx_ctx_mu);
    auto it = g_trx_ctx_map.find(owner);
    if (it != g_trx_ctx_map.end() && it->second.get() == ctx) {
      owned = std::move(it->second);
      g_trx_ctx_map.erase(it);
    }
  }

  if (!owned) {
    ctx->by_index.clear();
  }
}

// —— 收集一行放入全局tctx bucket中 ——
int vec_collect_one_row(trx_t *trx, dict_table_t *table, dict_index_t *vindex,
                        const dfield_t *vector_field, const unsigned dim,
                        const dtuple_t *row_tuple) {
  if (!trx || !table || !vindex || !vector_field || !row_tuple || dim == 0) {
    return -1;
  }

  vec_trx_ctx_t *tctx = vec_get_or_create_trx_ctx(trx);
  if (tctx == nullptr) {
    return -1;
  }

  auto &bucket = tctx->by_index[vindex];
  if (bucket.index == nullptr) {
    bucket.index = vindex;
    bucket.dim = dim;
  } else if (bucket.dim != dim) {
    return -4;  // dim 不一致
  }

  vec_item_t item;
  if (!vec_extract_and_validate(vector_field, dim, item.vec)) {
    return -2;  // 长度/数据非法
  }

  if (!vec_capture_pk_columns(table, row_tuple, item.pk_columns)) {
    return -3;
  }

  bucket.items.emplace_back(std::move(item));
  return 0;
}

// 放入已有的bucket中
int vec_collect_one_row(std::vector<vec_item_t> &bucket, dict_table_t *table, dict_index_t *vindex,
                        const dfield_t *vector_field, const unsigned dim,
                        const dtuple_t *row_tuple){

  if (!table || !vindex || !vector_field || !row_tuple || dim == 0) {
    return -1;
  }
  vec_item_t item;
  if (!vec_extract_and_validate(vector_field, dim, item.vec)) {
    return -2;  // 长度/数据非法
  }

  if (!vec_capture_pk_columns(table, row_tuple, item.pk_columns)) {
    return -3;
  }

  bucket.emplace_back(std::move(item));
  return 0;
}



vec_trx_ctx_t* vec_lookup_trx_ctx(trx_t* trx) {
    std::lock_guard<std::mutex> g(g_trx_ctx_mu);
    auto it = g_trx_ctx_map.find(trx);
    return it == g_trx_ctx_map.end() ? nullptr : it->second.get();
}

bool vec_trx_has_work(trx_t* trx) {
    if (auto* ctx = vec_lookup_trx_ctx(trx)) {
        for (const auto& kv : ctx->by_index) {
            if (!kv.second.items.empty()) return true;
        }
    }
    return false;
}
