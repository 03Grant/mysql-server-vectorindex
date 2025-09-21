#include "vec_txn_buf.h"

#include "dict0dict.h"
#include "mem0mem.h"
#include "row0row.h"
#include "ut0dbg.h"

#include "data0data.h"
#include "my_byteorder.h"
#include "sql/field.h"

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

inline void append_u16_be(std::string &buf, uint16_t v) {
  char tmp[2];
  tmp[0] = static_cast<char>((v >> 8) & 0xFF);
  tmp[1] = static_cast<char>(v & 0xFF);
  buf.append(tmp, sizeof(tmp));
}

inline void append_u32_be(std::string &buf, uint32_t v) {
  char tmp[4];
  tmp[0] = static_cast<char>((v >> 24) & 0xFF);
  tmp[1] = static_cast<char>((v >> 16) & 0xFF);
  tmp[2] = static_cast<char>((v >> 8) & 0xFF);
  tmp[3] = static_cast<char>(v & 0xFF);
  buf.append(tmp, sizeof(tmp));
}

// —— 主键编码：仿照 row_build_index_entry_low 得到聚簇索引 key ——
static bool vec_encode_pk_bin(dict_table_t *table, const dtuple_t *row,
                              std::string &out) {
  if (table == nullptr || row == nullptr) {
    return false;
  }

  dict_index_t *clust = table->first_index();
  if (clust == nullptr) {
    return false;
  }

  mem_heap_t *heap = mem_heap_create(512, UT_LOCATION_HERE);
  if (heap == nullptr) {
    return false;
  }

  dtuple_t *entry = row_build_index_entry_low(row, nullptr, clust, heap,
                                              ROW_BUILD_NORMAL);
  if (entry == nullptr) {
    mem_heap_free(heap);
    return false;
  }

  const ulint n_cmp = dtuple_get_n_fields_cmp(entry);
  if (n_cmp == ULINT_UNDEFINED || n_cmp > std::numeric_limits<uint16_t>::max()) {
    mem_heap_free(heap);
    return false;
  }

  size_t reserve = sizeof(uint16_t);
  for (ulint i = 0; i < n_cmp; ++i) {
    const dfield_t *df = dtuple_get_nth_field(entry, i);
    if (df == nullptr) {
      mem_heap_free(heap);
      return false;
    }
    reserve += 1;  // flag byte
    if (!dfield_is_null(df)) {
      const ulint len_ul = dfield_get_len(df);
      if (len_ul > std::numeric_limits<uint32_t>::max()) {
        mem_heap_free(heap);
        return false;
      }
      reserve += sizeof(uint32_t);
      reserve += static_cast<size_t>(len_ul);
    }
  }

  out.clear();
  out.reserve(reserve);
  append_u16_be(out, static_cast<uint16_t>(n_cmp));

  for (ulint i = 0; i < n_cmp; ++i) {
    const dfield_t *df = dtuple_get_nth_field(entry, i);
    const bool is_null = dfield_is_null(df);
    out.push_back(static_cast<char>(is_null ? 0x1 : 0x0));

    if (is_null) {
      continue;
    }

    const ulint len_ul = dfield_get_len(df);
    const uint32_t len = static_cast<uint32_t>(len_ul);
    append_u32_be(out, len);

    const void *data = dfield_get_data(df);
    if (len > 0 && data == nullptr) {
      mem_heap_free(heap);
      out.clear();
      return false;
    }
    out.append(reinterpret_cast<const char *>(data), len);
  }

  mem_heap_free(heap);
  return true;
}

// —— 抽取向量字节并校验 ——
static bool vec_extract_and_validate(Field *field, unsigned dim,
                                     std::vector<float> &out) {
  if (field == nullptr || dim == 0 || field->is_null()) {
    return false;
  }

  const size_t expect_bytes = static_cast<size_t>(dim) * sizeof(float);
  const unsigned char *raw = nullptr;
  size_t raw_len = 0;

  switch (field->real_type()) {
    case MYSQL_TYPE_VECTOR: {
      auto *vf = static_cast<Field_vector *>(field);
      raw = reinterpret_cast<const unsigned char *>(vf->get_blob_data());
      raw_len = vf->get_length();
      break;
    }
    case MYSQL_TYPE_VARCHAR:
    case MYSQL_TYPE_VAR_STRING: {
      if (!field->binary()) {
        return false;  // 只支持 VARBINARY
      }
      auto *vs = static_cast<Field_varstring *>(field);
      raw = reinterpret_cast<const unsigned char *>(vs->data_ptr());
      raw_len = vs->data_length();
      break;
    }
    default:
      return false;
  }

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

// —— 收集一行 ——
int vec_collect_one_row(trx_t *trx, dict_table_t *table, dict_index_t *vindex,
                        Field *vector_field, const unsigned dim,
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

  if (!vec_encode_pk_bin(table, row_tuple, item.pk_bin)) {
    return -3;
  }

  bucket.items.emplace_back(std::move(item));
  return 0;
}
