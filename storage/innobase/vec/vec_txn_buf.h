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

// Snapshot of one clustered primary key column, stored as raw bytes.
struct vec_pk_column_t {
  bool is_null{false};
  ulint mtype{0};
  ulint prtype{0};
  std::vector<unsigned char> data;
};

// DDL build path metadata for one deferred vector row.
struct vec_ddl_aux_row_t {
  std::vector<vec_pk_column_t> pk_columns;
  trx_id_t creator_trx_id{0};
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

// Transaction context attached to trx_t; may be stored via trx->user_thd().
struct vec_trx_ctx_t {
  trx_t* owner{nullptr};
  std::mutex mu;
  /**/
  // 2/18/2026 Not used now! Primary keys of delete requests collected in this transaction.
  std::unordered_map<dict_index_t*, std::unordered_set<std::string>>
      deleted_pks_in_trx;
  // Inserted vector IDs to mark on rollback.
  std::vector<vec_insert_undo_entry_t> inserted_vids;
  // Update rollback log entries (logging only).
  std::vector<vec_update_undo_entry_t> update_changes;
  // Indexes touched by immediate inserts (for commit-time checks).
  std::unordered_set<dict_index_t*> touched_indexes;
};

// ==== Public API ====

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

// Ensure the transaction has a reusable vec_trx_ctx.
vec_trx_ctx_t* vec_get_or_create_trx_ctx(trx_t* trx);

// Called on row insert/update: extract the vector and primary key snapshot, then write to the vector index and auxiliary tables immediately.
// Only validate length and dimensions; do not check metric preprocessing or modify bytes.
int vec_collect_one_row(trx_t*           trx,
                        dict_table_t*    table,
                        dict_index_t*    vindex,
                        const dfield_t*  vector_field,   // The dfield for this index's column
                        const unsigned   dim,
                        const dtuple_t*  row_tuple);     // The current row's InnoDB tuple

// DDL path: extract vector + PK only. The caller batches rows and flushes them
// into the mutable index later.
int vec_collect_one_row_no_aux(trx_t*           trx,
                               dict_table_t*    table,
                               dict_index_t*    vindex,
                               const dfield_t*  vector_field,
                               const unsigned   dim,
                               const dtuple_t*  row_tuple,
                               std::vector<float>* out_vec_values,
                               std::vector<vec_pk_column_t>* out_pk_columns,
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

// Batch insert into vector index + aux cache/table, used by DDL flush path.
dberr_t vec_insert_rows_no_aux(
    trx_t* trx,
    dict_table_t* table,
    dict_index_t* vindex,
    const float* xb,
    size_t n,
    const std::vector<vec_ddl_aux_row_t>& rows);

// Clear after a successful commit or rollback.
void vec_trx_ctx_clear(vec_trx_ctx_t* ctx);


vec_trx_ctx_t* vec_lookup_trx_ctx(trx_t* trx);
bool vec_trx_has_work(trx_t* trx);
