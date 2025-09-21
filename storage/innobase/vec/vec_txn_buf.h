#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>

struct trx_t;
struct dict_index_t;
struct dict_table_t;
struct Field;
struct dtuple_t;

#include "vec_index_runtime.h"   // vec_index_ctx_t / params
#include "vec_params.h"

// 一条待写入的数据（本事务内临时保存）
struct vec_item_t {
  std::string pk_bin;        // FTS 风格编码后的主键
  std::vector<float> vec;    // dim 个 float（不做任何预处理）
};

// 每个向量索引一个桶
struct vec_trx_bucket_t {
  dict_index_t* index{nullptr};        // 逻辑索引
  uint32_t      dim{0};
  std::vector<vec_item_t> items;       // 本事务要写的条目
};

// 事务级上下文（挂到 trx_t；你可以用 trx->user_thd() 侧链保存）
struct vec_trx_ctx_t {
  trx_t* owner{nullptr};
  // 以 dict_index_t* 为 key 的桶
  std::unordered_map<dict_index_t*, vec_trx_bucket_t> by_index;
};

// ==== 对外 API ====

// 保证事务上有一个 vec_trx_ctx，可复用
vec_trx_ctx_t* vec_get_or_create_trx_ctx(trx_t* trx);

// 在“插入/更新行”时调用：抽取向量 + 编码主键 + 放入桶
// 不检查度量前处理，不改动字节，只校验长度与维度。
int vec_collect_one_row(trx_t*           trx,
                        dict_table_t*    table,
                        dict_index_t*    vindex,
                        Field*           vector_field,   // 该索引对应的向量列
                        const unsigned   dim,
                        const dtuple_t*  row_tuple);     // 当前行的 InnoDB tuple

// 提交成功或回滚时清空
void vec_trx_ctx_clear(vec_trx_ctx_t* ctx);
