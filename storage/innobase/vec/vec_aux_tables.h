#pragma once
#include "dict0dict.h"
#include "dict0mem.h"
#include "row0mysql.h"
#include "mem0mem.h"
#include "ut0ut.h"

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