// vec_index_adapter.h
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include "vec_params.h"
#include "univ.i" 
#include "db0err.h"

struct vec_index_ctx_t;
struct dict_index_t;
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

// 创建/销毁（只 new，不加向量）
bool vec_create(vec_index_ctx_t& ctx, const vec_params_t& p);

// Drop some index in the context(maybe because of index merge), drop in_mem_index is not allowed unless allow_drop_mutable is true.
// I don't think we need to drop in_mem_index at any time.
bool vec_drop_index(vec_index_ctx_t& ctx, size_t seg_idx, bool allow_drop_mutable=false);


void vec_destroy(vec_index_ctx_t* ctx);

// 最小检索/插入（后面你再接入 InnoDB 行数据）
int vec_add(vec_index_ctx_t& ctx, const float* xb, size_t n);                   // 添加 n 向量

int vec_add_with_ids(vec_index_ctx_t& ctx, const float* xb, const int64_t* ids, size_t n);

int vec_search(vec_index_ctx_t& ctx, const float* q, size_t nq,
           size_t k, float* distances, int64_t* labels, uint32_t* segments);                   // 召回
