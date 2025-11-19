#include "vec_params.h"

// storage/innobase/vec/vec_params.cc
#include <algorithm>
#include <cctype>
#include <sstream>
#include <unordered_set>

namespace {

// 去两端空白
inline void trim(std::string& x) {
  auto not_space = [](int ch){ return !std::isspace(ch); };
  x.erase(x.begin(), std::find_if(x.begin(), x.end(), not_space));
  x.erase(std::find_if(x.rbegin(), x.rend(), not_space).base(), x.end());
}

// 转小写（就地）
inline void to_lower(std::string& x) {
  std::transform(x.begin(), x.end(), x.begin(),
                 [](unsigned char c){ return std::tolower(c); });
}

// 解析整数（支持十进制）。失败返回 false
template <typename IntT>
bool parse_int(const std::string& s, IntT& out) {
  if (s.empty()) return false;
  std::size_t pos = 0;
  try {
    long long v = 0;
    if constexpr (std::is_same_v<IntT, uint64_t>) {
      unsigned long long u = std::stoull(s, &pos, 10);
      if (pos != s.size()) return false;
      out = static_cast<uint64_t>(u);
      return true;
    } else {
      long long t = std::stoll(s, &pos, 10);
      if (pos != s.size()) return false;
      // 简单边界
      if constexpr (std::is_unsigned_v<IntT>) {
        if (t < 0) return false;
      }
      out = static_cast<IntT>(t);
      return true;
    }
  } catch (...) {
    return false;
  }
}

} // namespace

dberr_t vec_params_from_string(const std::string& s,
                               vec_params_t* out,
                               std::string* err)
{
  if (!out) return DB_FAIL;
  *out = vec_params_t{};  // 默认：type=FLAT, metric=L2, threads=16

  if (s.empty()) {
    if (err) *err = "empty parameter string";
    return DB_FAIL;  // 你要求 type/dim/size 必填
  }

  std::string line = s;
  for (char& c : line) {
    if (c == ';' ) c = ',';
  }

  bool seen_type = false, seen_dim = false, seen_size = false;
  bool seen_backend = false;

  std::stringstream ss(line);
  std::string item;
  while (std::getline(ss, item, ',')) {
    trim(item);
    if (item.empty()) continue;

    auto eq = item.find('=');
    if (eq == std::string::npos) continue;

    std::string key = item.substr(0, eq);
    std::string val = item.substr(eq + 1);
    trim(key); trim(val);

    std::string key_l = key; to_lower(key_l);
    std::string val_l = val; to_lower(val_l);

    if (key_l == "backend") {
      if (val_l == "faiss") out->backend = BackendType::Faiss;
      else if (val_l == "hnswlib" || val_l == "hnsw") out->backend = BackendType::Hnswlib;
      else {
        if (err) *err = "unsupported backend: " + val;
        return DB_UNSUPPORTED;
      }
      seen_backend = true;

    } else if (key_l == "type") {
      if      (val_l == "flat")    out->type_tag = VEC_T_FLAT;
      else if (val_l == "hnsw")    out->type_tag = VEC_T_HNSW;
      else if (val_l == "ivfflat") out->type_tag = VEC_T_IVFFLAT;
      else if (val_l == "ivfpq")   out->type_tag = VEC_T_IVFPQ;
      else {
        if (err) *err = "unsupported type: " + val;
        return DB_UNSUPPORTED;
      }
      seen_type = true;

    } else if (key_l == "dim") {
      uint32_t v{};
      if (!parse_int(val_l, v) || v == 0) {
        if (err) *err = "invalid dim: " + val;
        return DB_FAIL;
      }
      out->dim = v;
      seen_dim = true;

    } else if (key_l == "size") {
      uint64_t v{};
      if (!parse_int(val_l, v) || v < 0) {
        if (err) *err = "invalid size: " + val;
        return DB_FAIL;
      }
      out->size = v;
      seen_size = true;

    } else if (key_l == "dist" || key_l == "metric") {
      if (val_l == "l2") out->metric_tag = VEC_M_L2;
      else if (val_l == "ip" || val_l == "inner_product") out->metric_tag = VEC_M_IP;
      else if (val_l == "cosine") out->metric_tag = VEC_M_COSINE;
      else {
        if (err) *err = "invalid dist metric: " + val;
        return DB_FAIL;
      }

    } else if (key_l == "build_threads" || key_l == "threads") {
      int v{};
      if (!parse_int(val_l, v) || v <= 0) {
        if (err) *err = "invalid build_threads: " + val;
        return DB_FAIL;
      }
      out->build_threads = v;

    } else if (key_l == "ef") {
      int v{};
      if (!parse_int(val_l, v) || v <= 0) {
        if (err) *err = "invalid hnsw ef: " + val;
        return DB_FAIL;
      }
      out->efConstruction = v;

    } else if (key_l == "m") {
      int v{};
      if (!parse_int(val_l, v) || v <= 0) {
        if (err) *err = "invalid m: " + val;
        return DB_FAIL;
      }
      out->m = v;

    } else if (key_l == "nlist") {
      int v{};
      if (!parse_int(val_l, v) || v <= 0) {
        if (err) *err = "invalid nlist: " + val;
        return DB_FAIL;
      }
      out->nlist = v;

    } else if (key_l == "nbits" || key_l == "nbit") {
      int v{};
      if (!parse_int(val_l, v) || v <= 0) {
        if (err) *err = "invalid nbits: " + val;
        return DB_FAIL;
      }
      out->nbits = v;

    } else if (key_l == "hnsw_m") {
      int v{};
      if (!parse_int(val_l, v) || v <= 0) {
        if (err) *err = "invalid hnsw_m: " + val;
        return DB_FAIL;
      }
      out->hnsw_m = v;

    } else if (key_l == "nlist-search" || key_l == "nprobe") {
      // 先忽略（运行期搜索参数），不影响 DDL
    } else {
      // 未知键忽略（可选：累计 warning）
    }
  }

  // 必填校验
  if (!seen_type || !seen_dim || !seen_size) {
    if (err) {
      *err = "required fields missing:"
             + std::string(!seen_type ? " type" : "")
             + std::string(!seen_dim  ? " dim"  : "")
             + std::string(!seen_size ? " size" : "");
    }
    return DB_FAIL;
  }
  static_cast<void>(seen_backend);

  // 默认值
  if (out->build_threads <= 0) out->build_threads = 16;

  // 类型特定校验/默认
  switch (out->type_tag) {
    case VEC_T_FLAT:
      // 无附加要求
      break;

    case VEC_T_HNSW:
      if (out->efConstruction <= 0) out->efConstruction = 128;
      if (out->hnsw_m        <= 0) out->hnsw_m        = 16;
      break;

    case VEC_T_IVFFLAT:
      if (out->nlist <= 0) {
        if (err) *err = "ivfflat requires nlist > 0";
        return DB_FAIL;
      }
      break;

    case VEC_T_IVFPQ:
      if (out->nlist <= 0 || out->nbits <= 0 || out->m <= 0) {
        if (err) *err = "ivfpq requires nlist>0, nbits>0, m>0";
        return DB_FAIL;
      }
      {
        // 你要求：nbits * m 必须整除 dim
        const uint64_t prod = static_cast<uint64_t>(out->nbits) *
                              static_cast<uint64_t>(out->m);
        if (prod == 0 || (out->dim % prod) != 0) {
          if (err) {
            std::ostringstream os;
            os << "ivfpq requires (nbits * m) | dim, got nbits="
               << out->nbits << ", m=" << out->m << ", dim=" << out->dim;
            *err = os.str();
          }
          return DB_FAIL;
        }
      }
      break;

    default:
      if (err) *err = "unknown type tag";
      return DB_UNSUPPORTED;
  }

  return DB_SUCCESS;
}
