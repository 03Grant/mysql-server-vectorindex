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

struct vec_index_ctx_t;

// Record inserted vector IDs for rollback cleanup.
struct vec_insert_undo_entry_t {
  vec_index_ctx_t* ctx{nullptr};
  dict_index_t*    index{nullptr};
  std::string      segment_id;
  uint64_t         vid{0};
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
  /**/
  // 2/18/2026 Not used now! 删除请求的 pk（在本事务内收集）
  std::unordered_map<dict_index_t*, std::unordered_set<std::string>>
      deleted_pks_in_trx;
  // Inserted vector IDs to mark on rollback.
  std::vector<vec_insert_undo_entry_t> inserted_vids;
  // Update rollback log entries (logging only).
  std::vector<vec_update_undo_entry_t> update_changes;
  // Indexes touched by immediate inserts (for commit-time checks).
  std::unordered_set<dict_index_t*> touched_indexes;
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

// 在“插入/更新行”时调用：抽取向量 + 主键快照 + 立即写入向量索引与辅助表
// 不检查度量前处理，不改动字节，只校验长度与维度。
int vec_collect_one_row(trx_t*           trx,
                        dict_table_t*    table,
                        dict_index_t*    vindex,
                        const dfield_t*  vector_field,   // 该索引对应列的 dfield
                        const unsigned   dim,
                        const dtuple_t*  row_tuple);     // 当前行的 InnoDB tuple

// DDL path: extract vector + PK, insert into index and aux cache only.
// Returns pk columns + seg_id for deferred aux table insert.
int vec_collect_one_row_no_aux(trx_t*           trx,
                               dict_table_t*    table,
                               dict_index_t*    vindex,
                               const dfield_t*  vector_field,
                               const unsigned   dim,
                               const dtuple_t*  row_tuple,
                               std::vector<vec_pk_column_t>* out_pk_columns,
                               std::string*     out_seg_id,
                               trx_id_t         creator_trx_id = 0);

// Immediate insert path: add to vector index and aux table, record rollback info.
dberr_t vec_insert_one_row(trx_t* trx,
                           dict_table_t* table,
                           dict_index_t* vindex,
                           const std::vector<vec_pk_column_t>& pk_columns,
                           const std::vector<float>& vec_values,
                           uint64_t* out_vid);

// Insert into vector index + aux cache, skip aux table insert (DDL use).
dberr_t vec_insert_one_row_no_aux(
    trx_t* trx,
    dict_table_t* table,
    dict_index_t* vindex,
    const std::vector<vec_pk_column_t>& pk_columns,
    const std::vector<float>& vec_values,
    uint64_t* out_vid,
    std::string* out_seg_id,
    trx_id_t creator_trx_id = 0);

// 提交成功或回滚时清空
void vec_trx_ctx_clear(vec_trx_ctx_t* ctx);


vec_trx_ctx_t* vec_lookup_trx_ctx(trx_t* trx);
bool vec_trx_has_work(trx_t* trx);
