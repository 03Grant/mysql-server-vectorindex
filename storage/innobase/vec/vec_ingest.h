#pragma once
#include "univ.i"
#include "vec_txn_buf.h"

dberr_t vec_on_trx_commit(trx_t* trx);
void vec_on_trx_rollback(trx_t* trx);