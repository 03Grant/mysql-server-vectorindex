#pragma once

#include "db0err.h"
#include <string>
#include <optional>

enum class BackendType {
    Faiss,
    Hnswlib,
    Diskann
};

enum : uint8_t {
  VEC_T_FLAT    = 0,
  VEC_T_HNSW    = 1,
  VEC_T_IVFFLAT = 2,
  VEC_T_IVFPQ   = 3
};

enum : uint8_t {
  VEC_M_L2     = 0,
  VEC_M_IP     = 1,  // inner_product
  VEC_M_COSINE = 2
};

// 纯 POD，适合放到 mem_heap 里；不要 std::string
struct vec_params_t {
  BackendType backend{BackendType::Faiss};
  uint8_t   type_tag{VEC_T_FLAT};   // 上面枚举
  uint8_t   metric_tag{VEC_M_L2};   // 上面枚举
  uint32_t  dim{0};
  uint64_t  size{0};             // Size to flush, build immutable index when exceeded

  int32_t   build_threads{16};

  // IVF / PQ / HNSW
  int32_t   nlist{0};        // IVF
  int32_t   m{0}, nbits{0};  // PQ
  int32_t   hnsw_m{32}, efConstruction{128}; // HNSW
};
// 简单占位（你后面可改成从 DD / JSON 解析）
dberr_t vec_params_from_string(const std::string& s,
                               vec_params_t* out,
                               std::string* err = nullptr);
