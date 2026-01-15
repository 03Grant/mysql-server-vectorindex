#pragma once
#include "univ.i"
#include <cstddef>
#include <vector>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

struct trx_t;
struct dict_index_t;
struct dict_table_t;
struct dfield_t;
struct dtuple_t;

#include "vec_index_runtime.h"   // vec_index_ctx_t / params
#include "vec_params.h"

// 聚簇主键的单列快照（按原始字节存储）
struct vec_pk_column_t {
  bool is_null{false};
  ulint mtype{0};
  ulint prtype{0};
  std::vector<unsigned char> data;
};

// 一条待写入的数据（本事务内临时保存）
struct vec_item_t {
  std::vector<vec_pk_column_t> pk_columns;  // 聚簇主键逐列快照
  std::vector<float>           vec;         // dim 个 float（不做任何预处理）
  std::string                  pk_key;      // packed primary key bytes
};

enum class vec_aux_mode_t : uint8_t {
  UNKNOWN = 0,
  PREINSERT_NULL,   // DML path: PK inserted with NULL faiss_id during collect
  DIRECT_INSERT     // DDL/builder path: insert final rows during commit
};

// 每个向量索引一个桶
struct vec_trx_bucket_t {
  dict_index_t* index{nullptr};        // 逻辑索引
  uint32_t      dim{0};
  vec_aux_mode_t aux_mode{vec_aux_mode_t::UNKNOWN};
  std::vector<vec_item_t> items;       // 本事务要写的条目
  std::unordered_set<std::string> inserted_keys; // dedupe pending inserts
};

struct vec_index_ctx_t;

// 记录 bitmap 改动，供回滚恢复
struct vec_bitmap_undo_entry_t {
  vec_index_ctx_t* ctx{nullptr};
  dict_index_t*    index{nullptr};
  uint32_t         segment_id{0};
  uint64_t         vid{0};
  bool             old_val{false};
};

// Record update details for rollback logging.
struct vec_update_undo_entry_t {
  dict_index_t* index{nullptr};
  std::string   old_pk_key;
  std::string   new_pk_key;
  bool          in_bucket{false};
};

// 事务级上下文（挂到 trx_t；你可以用 trx->user_thd() 侧链保存）
struct vec_trx_ctx_t {
  trx_t* owner{nullptr};
  std::mutex mu;
  // 以 dict_index_t* 为 key 的桶
  std::unordered_map<dict_index_t*, vec_trx_bucket_t> by_index;
  // 删除请求的 pk（在本事务内收集）
  std::unordered_map<dict_index_t*, std::unordered_set<std::string>>
      deleted_pks_in_trx;
  // 提交前 bitmap 改动的回滚日志
  std::vector<vec_bitmap_undo_entry_t> bitmap_changes;
  // Update rollback log entries (logging only).
  std::vector<vec_update_undo_entry_t> update_changes;
};

// ==== 对外 API ====

bool vec_capture_pk_columns(dict_table_t* table,
                            const dtuple_t* row,
                            std::vector<vec_pk_column_t>& out);

std::string vec_pack_pk_key(const std::vector<vec_pk_column_t>& pk_columns,
                            ulint pk_fields);

std::string vec_pack_pk_key_from_tuple(dict_table_t* table,
                                       const dtuple_t* row,
                                       ulint pk_fields);

std::string vec_format_pk_columns_debug(
    const std::vector<vec_pk_column_t>& cols,
    ulint pk_fields,
    size_t preview_bytes = 32);

// 保证事务上有一个 vec_trx_ctx，可复用
vec_trx_ctx_t* vec_get_or_create_trx_ctx(trx_t* trx);

// 在“插入/更新行”时调用：抽取向量 + 主键快照 + 放入桶
// 不检查度量前处理，不改动字节，只校验长度与维度。
int vec_collect_one_row(trx_t*           trx,
                        dict_table_t*    table,
                        dict_index_t*    vindex,
                        const dfield_t*  vector_field,   // 该索引对应列的 dfield
                        const unsigned   dim,
                        const dtuple_t*  row_tuple);     // 当前行的 InnoDB tuple

int vec_collect_one_row(std::vector<vec_item_t> &bucket, 
                        dict_table_t*   table, 
                        dict_index_t*   vindex,
                        const dfield_t* vector_field,
                        const unsigned  dim,
                        const dtuple_t* row_tuple);

// 提交成功或回滚时清空
void vec_trx_ctx_clear(vec_trx_ctx_t* ctx);


vec_trx_ctx_t* vec_lookup_trx_ctx(trx_t* trx);
bool vec_trx_has_work(trx_t* trx);
