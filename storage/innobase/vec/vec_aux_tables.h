#pragma once
#include <vector>
#include "dict0dict.h"
#include "dict0mem.h"
#include "row0mysql.h"
#include "mem0mem.h"
#include "ut0ut.h"
#include "pars0pars.h"

#include "vec_txn_buf.h"

struct trx_t;
struct dict_index_t;

/** Create auxiliary index tables for an FTS index.
@param[in,out]  trx             transaction
@param[in]      index           the index instance
@param[in]      table_name      table name
@param[in]      table_id        the table id
@return DB_SUCCESS or error code */
dberr_t vec_create_index_tables_low(trx_t *trx, dict_index_t *index,
                                    const char *table_name,
                                    table_id_t table_id);

dberr_t vec_create_index_dd_tables(dict_table_t *table);


struct vec_pk_column_t;

dberr_t vec_aux_insert_one(trx_t* trx,
                           dict_index_t* index,
                           const std::vector<vec_pk_column_t>& pk_columns,
                           uint64_t faiss_id);


que_t* vec_parse_sql(const char* table_name_or_null, pars_info_t* info, const char* sql_body); 

dberr_t vec_eval_sql(trx_t* trx, que_t* graph);