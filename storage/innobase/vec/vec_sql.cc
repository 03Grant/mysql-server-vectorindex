#include <sys/types.h>

#include "dict0dd.h"
#include "dict0dict.h"
#include "pars0pars.h"
#include "que0que.h"
#include "trx0roll.h"
#include "ut0mutex.h"
#include "vec_aux_tables.h"

#include <algorithm>
#include <string>
#include "current_thd.h"


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

  // 1) 包装 SQL
  char* full = ut_str3cat(vec_sql_begin, sql_body, vec_sql_end); // 或加你自己的 end

  // 2) 预开表（可选，但强烈推荐）
  if (table_name_or_null) {
    pre = dd_table_open_on_name_in_mem(table_name_or_null, false);
    if (!pre) {
      pre = dd_table_open_on_name(thd, &mdl, table_name_or_null, false, DICT_ERR_IGNORE_NONE);
    }
  }

  // 3) 解析（串行化）
  mutex_enter(&pars_mutex);
  que_t* graph = pars_sql(info, full);
  mutex_exit(&pars_mutex);

  // 4) 收尾
  if (pre) dd_table_close(pre, thd, &mdl, false);
  ut::free(full);
  return graph; // 之后用 fts_eval_sql(trx, graph) 执行；不用时 que_graph_free(graph)
}


/** Execute a prepared que graph within a given transaction.
    Returns DB_SUCCESS or an error code. */
dberr_t vec_eval_sql(trx_t* trx, que_t* graph) {
  ut_ad(graph != nullptr);
  ut_ad(trx != nullptr);               // 调用方负责提供事务

  graph->trx = trx;                    // 用哪个事务执行
  graph->fork_type = QUE_FORK_MYSQL_INTERFACE;

  que_thr_t* thr = que_fork_start_command(graph);
  ut_a(thr);                           // 入口线程必须建好

  que_run_threads(thr);                // 同步执行直至完成/报错
  return trx->error_state;             // 成败看事务的 error_state
}