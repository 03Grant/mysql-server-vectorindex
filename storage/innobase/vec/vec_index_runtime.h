// vec_index_runtime.h
#pragma once
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

struct vec_index_ctx_t {
  std::mutex               mu;
  std::vector<std::unique_ptr<faiss::Index>> indices;  // runtime handler, include one in_mem_index(IVFFLAT), and several immutable indexes.
  vec_id_allocator_t        id_alloc;    // monotonic id allocator for Faiss IDs
  vec_params_t              params;      // parameters for immutable index not for in_mem_index
  bool                      inited{false}; // inited is true when in_mem_index is created successfully
  struct dict_table_t       *aux_dict_table{nullptr};
};
