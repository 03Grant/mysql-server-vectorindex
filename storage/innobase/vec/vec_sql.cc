#include <sys/types.h>

#include "db0err.h"
#include "dict0dd.h"
#include "dict0dict.h"
#include "pars0pars.h"
#include "que0que.h"
#include "trx0roll.h"
#include "ut0mutex.h"
#include "vec_aux_tables.h"

#include <algorithm>
#include <chrono>
#include <string>
#include <thread>
#include "current_thd.h"

namespace {

constexpr int kVecSqlLockWaitRetries = 2;
constexpr std::chrono::milliseconds kVecSqlRetryBase{50};
constexpr std::chrono::milliseconds kVecSqlRetryMax{500};

std::chrono::milliseconds vec_lock_wait_backoff(int retry_index) {
  const int shift = std::min(retry_index, 4);
  auto sleep = kVecSqlRetryBase * (1 << shift);
  if (sleep > kVecSqlRetryMax) {
    sleep = kVecSqlRetryMax;
  }
  return sleep;
}

} // namespace

/** Preamble to all SQL statements. */
static const char *vec_sql_begin = "PROCEDURE P() IS\n";

/** Postamble to non-committing SQL statements. */
static const char *vec_sql_end =
    "\n"
    "END;\n";


que_t* vec_parse_sql(const char* table_name_or_null,
                     pars_info_t* info,
                     const char* sql_body) {
  THD* thd = current_thd;
  MDL_ticket* mdl = nullptr;
  dict_table_t* pre = nullptr;

  // 1) Wrap the SQL.
  char* full = ut_str3cat(vec_sql_begin, sql_body, vec_sql_end); // Or supply a custom SQL suffix.

  // 2) Pre-open tables (optional but strongly recommended).
  if (table_name_or_null) {
    pre = dd_table_open_on_name_in_mem(table_name_or_null, false);
    if (!pre) {
      pre = dd_table_open_on_name(thd, &mdl, table_name_or_null, false, DICT_ERR_IGNORE_NONE);
    }
  }

  // 3) Parse with serialized access.
  mutex_enter(&pars_mutex);
  que_t* graph = pars_sql(info, full);
  mutex_exit(&pars_mutex);

  // 4) Clean up.
  if (pre) dd_table_close(pre, thd, &mdl, false);
  ut::free(full);
  return graph; // Execute with fts_eval_sql(trx, graph); release with que_graph_free(graph).
}


/** Execute a prepared que graph within a given transaction.
    Returns DB_SUCCESS or an error code. */
dberr_t vec_eval_sql(trx_t* trx, que_t* graph) {
  ut_ad(graph != nullptr);
  ut_ad(trx != nullptr);               // The caller must provide a transaction.

  if (trx->error_state != DB_SUCCESS &&
      trx->error_state != DB_LOCK_WAIT_TIMEOUT) {
    return trx->error_state;
  }

  dberr_t err = DB_SUCCESS;
  for (int retry = 0; retry <= kVecSqlLockWaitRetries; ++retry) {
    if (retry > 0) {
      std::this_thread::sleep_for(vec_lock_wait_backoff(retry - 1));
    }

    trx->error_state = DB_SUCCESS;

    graph->trx = trx;                  // Set the transaction used for execution.
    graph->fork_type = QUE_FORK_MYSQL_INTERFACE;

    que_thr_t* thr = que_fork_start_command(graph);
    ut_a(thr);                         // The entry thread must exist.

    que_run_threads(thr);              // Run synchronously until completion or error.
    err = trx->error_state;
    if (err != DB_LOCK_WAIT_TIMEOUT) {
      break;
    }
  }

  return err;                          // The transaction's error_state determines success or failure.
}
