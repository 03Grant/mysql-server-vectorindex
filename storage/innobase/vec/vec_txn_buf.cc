#include "vec_txn_buf.h"

#include "dict0dict.h"
#include "trx0trx.h"
#include "vec_aux_tables.h"
#include "vec_tasks.h"
#include "ut0log.h"

#include "data0data.h"
#include "my_byteorder.h"
#include "my_sys.h"
#include "mysqld_error.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <sstream>
#include <iomanip>

namespace {

std::mutex g_trx_ctx_mu;
std::unordered_map<trx_t *, std::unique_ptr<vec_trx_ctx_t>> g_trx_ctx_map;

inline ulint vec_pk_field_count(const dict_index_t *clust) {
  if (clust == nullptr) {
    return 0;
  }
  if (clust->name != nullptr &&
      std::strcmp(clust->name, "GEN_CLUST_INDEX") == 0) {
    return 1;
  }
  return clust->n_uniq;
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

// —— 聚簇主键逐列拷贝 ——
bool vec_capture_pk_columns(dict_table_t *table, const dtuple_t *row,
                            std::vector<vec_pk_column_t> &out) {
  if (table == nullptr || row == nullptr) {
    return false;
  }

  dict_index_t *clust = table->first_index();   // get clustered index
  const ulint pk_fields = vec_pk_field_count(clust);
  if (clust == nullptr || pk_fields == 0) {
    return false;
  }

  const ulint row_fields = dtuple_get_n_fields(row);
  if (row_fields == 0) {
    return false;
  }

  out.clear();
  out.reserve(pk_fields);

  for (ulint i = 0; i < pk_fields; ++i) {
    const dict_field_t *ind_field = clust->get_field(i);
    if (ind_field != nullptr && ind_field->col != nullptr) {
      // For clust_ref tuples, column numbers may be larger than row_fields,
      // so fall back to index position when needed.
      const ulint col_no = dict_col_get_no(ind_field->col);
      const dfield_t *df = nullptr;

      if (col_no < row_fields) {
        df = dtuple_get_nth_field(row, col_no);
      } else if (i < row_fields) {
        df = dtuple_get_nth_field(row, i);
      }

      if (df == nullptr) {
        ib::warn() << "vec_capture_pk_columns: dfield null"
                   << " i=" << i
                   << " col_no=" << col_no
                   << " row_fields=" << row_fields;
        out.clear();
        return false;
      }

      vec_pk_column_t col;
      col.is_null = dfield_is_null(df);
      col.mtype = ind_field->col->mtype;
      col.prtype = ind_field->col->prtype;

      if (!col.is_null) {
        const unsigned char *data =
            static_cast<const unsigned char *>(dfield_get_data(df));
        const ulint len = dfield_get_len(df);

        if (len == UNIV_SQL_NULL) {
          ib::warn() << "vec_capture_pk_columns: len is UNIV_SQL_NULL"
                     << " i=" << i
                     << " col_no=" << col_no;
          col.is_null = true;
        } else {
          if (len > 0 && data == nullptr) {
            ib::warn() << "vec_capture_pk_columns: data null with len>0"
                       << " i=" << i
                       << " col_no=" << col_no
                       << " len=" << len;
            out.clear();
            return false;
          }
          col.data.assign(data, data + len);
        }
      } else {
        ib::warn() << "vec_capture_pk_columns: captured NULL pk column"
                   << " i=" << i
                   << " col_no=" << col_no;
      }

      out.emplace_back(std::move(col));
    } else {
      ib::warn() << "vec_capture_pk_columns: missing index field"
                 << " i=" << i
                 << " col_no=" << (ind_field ? dict_col_get_no(ind_field->col) : ULINT_UNDEFINED);
      out.clear();
      return false;
    }
  }

  return true;
}

namespace {

inline void vec_pack_u32(std::string &buf, uint32_t value) {
  buf.push_back(static_cast<char>(value & 0xFF));
  buf.push_back(static_cast<char>((value >> 8) & 0xFF));
  buf.push_back(static_cast<char>((value >> 16) & 0xFF));
  buf.push_back(static_cast<char>((value >> 24) & 0xFF));
}

}  // namespace

std::string vec_pack_pk_key(const std::vector<vec_pk_column_t>& pk_columns,
                            ulint pk_fields) {
  const ulint cols = std::min<ulint>(pk_fields, pk_columns.size());
  const uint32_t num_cols = static_cast<uint32_t>(cols);

  size_t total = sizeof(uint32_t);
  for (ulint i = 0; i < cols; ++i) {
    total += sizeof(uint32_t);
    if (!pk_columns[i].is_null) {
      total += pk_columns[i].data.size();
    }
  }

  std::string packed;
  packed.reserve(total);
  vec_pack_u32(packed, num_cols);

  for (ulint i = 0; i < cols; ++i) {
    const auto &col = pk_columns[i];
    if (col.is_null) {
      vec_pack_u32(packed, std::numeric_limits<uint32_t>::max());
      continue;
    }
    const uint32_t len =
        static_cast<uint32_t>(std::min<size_t>(col.data.size(), UINT32_MAX));
    vec_pack_u32(packed, len);
    packed.append(reinterpret_cast<const char *>(col.data.data()), len);
  }

  return packed;
}

std::string vec_pack_pk_key_from_tuple(dict_table_t* table,
                                       const dtuple_t* row,
                                       ulint pk_fields) {
  std::vector<vec_pk_column_t> pk_columns;
  if (!vec_capture_pk_columns(table, row, pk_columns)) {
    return {};
  }
  return vec_pack_pk_key(pk_columns, pk_fields);
}

std::string vec_format_pk_columns_debug(const std::vector<vec_pk_column_t>& cols,
                                        ulint pk_fields,
                                        size_t preview_bytes) {
  std::ostringstream os;
  os << "pk_fields=" << pk_fields << " cols=" << cols.size();
  const ulint limit =
      pk_fields == 0 ? static_cast<ulint>(cols.size())
                     : std::min<ulint>(pk_fields, cols.size());
  for (ulint i = 0; i < limit; ++i) {
    const auto& c = cols[i];
    os << " [i=" << i << " null=" << (c.is_null ? 1 : 0)
       << " mtype=" << c.mtype << " prtype=" << c.prtype;
    if (!c.is_null) {
      const size_t len = c.data.size();
      const size_t preview = std::min(preview_bytes, len);
      os << " len=" << len << " hex=";
      for (size_t j = 0; j < preview; ++j) {
        os << std::hex << std::setw(2) << std::setfill('0')
           << static_cast<unsigned>(c.data[j]);
      }
      if (len > preview) {
        os << "...";
      }
      os << std::dec << std::setfill(' ');

      bool printable = true;
      for (size_t j = 0; j < std::min<size_t>(preview, 32); ++j) {
        if (!std::isprint(static_cast<unsigned char>(c.data[j]))) {
          printable = false;
          break;
        }
      }
      if (printable && preview > 0) {
        os << " ascii=\"";
        for (size_t j = 0; j < std::min<size_t>(preview, size_t{32}); ++j) {
          unsigned char ch = c.data[j];
          if (!std::isprint(ch)) {
            os << '?';
            break;
          }
          os << static_cast<char>(ch);
        }
        if (len > preview) {
          os << "...";
        }
        os << "\"";
      }
    }
    os << "]";
  }
  if (cols.size() < pk_fields) {
    os << " [missing_cols=" << (pk_fields - cols.size()) << "]";
  }
  return os.str();
}

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

  {
    std::lock_guard<std::mutex> lk(ctx->mu);
    for (auto &kv : ctx->by_index) {
      kv.second.items.clear();
      kv.second.inserted_keys.clear();
      kv.second.aux_mode = vec_aux_mode_t::UNKNOWN;
    }
    ctx->deleted_pks_in_trx.clear();
    ctx->bitmap_changes.clear();
    ctx->update_changes.clear();
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

  dict_index_t* clust = table->first_index();
  const ulint pk_fields = vec_pk_field_count(clust);
  if (clust == nullptr || pk_fields == 0) {
    trx->error_state = DB_ERROR;
    return -1;
  }

  vec_index_ctx_t *ctx = vindex->vec_runtime;
  if (ctx != nullptr) {
    const VecBootstrapState state =
        ctx->bootstrap_state.load(std::memory_order_acquire);
    if (state != VecBootstrapState::READY) {
      if (state == VecBootstrapState::NOT_STARTED ||
          state == VecBootstrapState::FAILED) {
        vec_schedule_bootstrap_load(vindex);
      }
      my_error(ER_INTERNAL_ERROR, MYF(0), kVecIndexLoadingMsg);
      trx->error_state = DB_VECINDEX_NOT_READY;
      return -7;
    }
  }

  vec_trx_ctx_t *tctx = vec_get_or_create_trx_ctx(trx);
  if (tctx == nullptr) {
    return -1;
  }

  vec_item_t item;
  if (!vec_extract_and_validate(vector_field, dim, item.vec)) {
    trx->error_state = DB_ERROR;
    return -2;  // 长度/数据非法
  }

  if (!vec_capture_pk_columns(table, row_tuple, item.pk_columns)) {
    trx->error_state = DB_ERROR;
    return -3;
  }

  item.pk_key = vec_pack_pk_key(item.pk_columns, pk_fields);
  if (item.pk_key.empty()) {
    trx->error_state = DB_ERROR;
    return -3;
  }

  dberr_t aux_err = vec_aux_insert_pk_null(trx, vindex, item.pk_columns);
  if (aux_err != DB_SUCCESS) {
    trx->error_state = aux_err;
    return -5;
  }

  {
    std::lock_guard<std::mutex> lk(tctx->mu);
    auto &bucket = tctx->by_index[vindex];
    if (bucket.index == nullptr) {
      bucket.index = vindex;
      bucket.dim = dim;
      bucket.aux_mode = vec_aux_mode_t::PREINSERT_NULL;
    } else if (bucket.dim != dim) {
      trx->error_state = DB_ERROR;
      return -4;  // dim 不一致
    }

    if (bucket.aux_mode == vec_aux_mode_t::UNKNOWN) {
      bucket.aux_mode = vec_aux_mode_t::PREINSERT_NULL;
    } else if (bucket.aux_mode != vec_aux_mode_t::PREINSERT_NULL) {
      if (bucket.items.empty()) {
        bucket.aux_mode = vec_aux_mode_t::PREINSERT_NULL;
      } else {
        trx->error_state = DB_ERROR;
        return -6;
      }
    }

    bucket.inserted_keys.insert(item.pk_key);
    bucket.items.emplace_back(std::move(item));
  }
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
