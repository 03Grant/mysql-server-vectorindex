// vec_index_adapter.h
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include "vec_params.h"
#include "univ.i"
#include "db0err.h"
#include "trx0types.h"

struct vec_index_ctx_t;
struct vec_index_version_t;
struct dict_index_t;
struct VecRuntimeSearchParams;
class THD;
struct TABLE;
class handler;
class MDL_ticket;

struct vec_aux_table_handle {
  TABLE *table{nullptr};
  handler *se_handler{nullptr};
  MDL_ticket *mdl_ticket{nullptr};
};

dberr_t vec_create_index_low(dict_index_t* idx);
dberr_t vec_open_aux_table(dict_index_t* idx);
dberr_t vec_open_aux_table_for_thd(dict_index_t* idx, THD *thd,
                                   vec_aux_table_handle *handle);
void vec_close_aux_table_for_thd(THD *thd, vec_aux_table_handle *handle);

// Create/destroy the index; creation does not add vectors.
bool vec_create(vec_index_ctx_t& ctx, const vec_params_t& p);

// Drop some index in the context(maybe because of index merge), drop in_mem_index is not allowed unless allow_drop_mutable is true.
// I don't think we need to drop in_mem_index at any time.
bool vec_drop_index(vec_index_ctx_t& ctx, size_t seg_idx, bool allow_drop_mutable=false);


void vec_destroy(vec_index_ctx_t* ctx);

// Basic search/insertion API; InnoDB row integration can be added later.
int vec_add(vec_index_ctx_t& ctx, const float* xb, size_t n);                   // Add n vectors

int vec_add_with_ids(vec_index_ctx_t& ctx, const float* xb, const int64_t* ids, size_t n);

// Search over a pinned, immutable version snapshot `ver` (see vec_pin_version).
// The caller must keep `ver` pinned for the lifetime of the call and of any
// follow-up that dereferences the returned segment ids.
int vec_search(vec_index_ctx_t& ctx, const vec_index_version_t& ver,
           const float* q, size_t nq,
           size_t k, float* distances, int64_t* labels, std::string* segments,
           trx_id_t* trx_ids,
           const VecRuntimeSearchParams* params = nullptr);                   // Search
