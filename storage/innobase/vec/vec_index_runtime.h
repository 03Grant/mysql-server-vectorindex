// vec_index_runtime.h
#pragma once
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>
namespace faiss { struct Index; }
#include "vec_id_alloc.h"
#include "vec_params.h"

struct dict_table_t;
struct TABLE;
class handler;
class MDL_ticket;

struct vec_index_aux_cache_t {
  size_t key_length{0};  // length of MySQL-format PK tuple
  std::vector<std::vector<unsigned char>> pk_values;  // indexed by faiss_id
  bool ready{false};

  void clear() {
    key_length = 0;
    pk_values.clear();
    ready = false;
  }

  size_t size() const { return pk_values.size(); }

  const unsigned char *get(size_t idx) const {
    if (!ready || idx >= pk_values.size() || pk_values[idx].empty()) {
      return nullptr;
    }
    return pk_values[idx].data();
  }
};

struct vec_index_ctx_t {
  std::mutex               mu;
  std::vector<std::unique_ptr<faiss::Index>> indices;  // runtime handler, include one in_mem_index(IVFFLAT), and several immutable indexes.
  vec_id_allocator_t        id_alloc;    // monotonic id allocator for Faiss IDs
  vec_params_t              params;      // parameters for immutable index not for in_mem_index
  bool                      inited{false}; // inited is true when in_mem_index is created successfully
  struct dict_table_t       *aux_dict_table{nullptr};
  vec_index_aux_cache_t      aux_cache;
};
