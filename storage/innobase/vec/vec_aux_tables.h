#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include "dict0dict.h"
#include "dict0mem.h"
#include "row0mysql.h"
#include "mem0mem.h"
#include "ut0ut.h"
#include "pars0pars.h"

#include "vec_txn_buf.h"

struct trx_t;
struct dict_index_t;
struct dtuple_t;
struct vid_pk_mapping_t;
struct TABLE;
struct KEY;
class Field;
class handler;

/** Create auxiliary index tables for a vecindex.
@param[in,out]  trx             transaction
@param[in]      index           the index instance
@param[in]      table_name      table name
@param[in]      table_id        the table id
@return DB_SUCCESS or error code */
dberr_t vec_create_index_tables_low(trx_t *trx, dict_index_t *index,
                                    const char *table_name,
                                    table_id_t table_id);

// Naming helpers for segment-specific auxiliary tables.
// Base prefix: "<db>/I_VEC_<table_id>_<index_id>"
std::string vec_aux_prefix(const dict_index_t *index);
std::string vec_aux_active_name(const dict_index_t *index);
std::string vec_aux_segment_name(const std::string& prefix, uint32_t seg_id);
std::string vec_aux_mem_name(const std::string& prefix);
bool vec_aux_extract_seg_id(const std::string& full_name,
                            const std::string& prefix,
                            uint32_t* seg_id_out);
bool vec_aux_table_exists(const std::string& full_name);
uint32_t vec_aux_scan_max_segment(const std::string& prefix,
                                  uint32_t probe_limit = 10000);
dberr_t vec_aux_rename_table(trx_t* trx,
                             const std::string& old_name,
                             const std::string& new_name);
dberr_t vec_aux_create_table(trx_t* trx,
                             dict_index_t* index,
                             const std::string& full_name);

dberr_t vec_create_index_dd_tables(dict_table_t *table);

// Append pk snapshot into cache
dberr_t vec_insert_aux_cache(vid_pk_mapping_t *cache,
                             dict_index_t *clust_index, uint64_t faiss_id,
                             const std::vector<vec_pk_column_t> &pk_columns);

// Bind cached PK entry directly to a clustered-index tuple
bool vec_aux_cache_bind_tuple(const vid_pk_mapping_t *cache,
                              uint64_t faiss_id, dict_index_t *clust_index,
                              dtuple_t *tuple);

// Persist/restore vid->PK mapping snapshots alongside immutable vector index files.
std::string vec_vid_pk_mapping_path(const std::string& index_path);
bool vec_vid_pk_mapping_save(const vid_pk_mapping_t& mapping,
                             const std::string& path);
bool vec_vid_pk_mapping_load(const std::string& path,
                             vid_pk_mapping_t* mapping);


struct vec_pk_column_t;

dberr_t vec_aux_insert_one(trx_t* trx,
                           dict_index_t* index,
                           const std::vector<vec_pk_column_t>& pk_columns,
                           uint64_t faiss_id);

dberr_t vec_aux_insert_pk_null(trx_t* trx,
                               dict_index_t* index,
                               const std::vector<vec_pk_column_t>& pk_columns);

dberr_t vec_aux_update_pk_vid(trx_t* trx,
                              dict_index_t* index,
                              const std::vector<vec_pk_column_t>& pk_columns,
                              uint64_t faiss_id);

que_t* vec_parse_sql(const char* table_name_or_null, pars_info_t* info, const char* sql_body); 

dberr_t vec_eval_sql(trx_t* trx, que_t* graph);
