#pragma once

#include <cstdint>
#include <mutex>
#include <limits>
#include "db0err.h"
#include "trx0trx.h"


// 前置声明（来自 InnoDB）
struct trx_t;
struct dict_index_t;

/**
 * 向量索引的单调 ID 分配器（每个 dict_index_t 一份）
 * - 懒恢复：首次调用 recover_from_aux() 时，从 aux 表读 MAX(faiss_id)。
 * - 分配：reserve(k, &start) 预留一段 [start, start+k-1]。
 */
struct vec_id_allocator_t {
  // 运行态
  std::mutex mu{};
  uint64_t   next{0};
  bool       inited{false};

  // === 生命周期 ===

  /**
   * 懒恢复：从辅助表读取 MAX(faiss_id)，把 next 设为 max+1。
   * 可多次调用，内部只会在未初始化时生效。
   *
   * @param trx   事务句柄（读 aux 表用）
   * @param index 哪个向量索引（用于定位 aux 表名）
   * @return DB_SUCCESS / 其他错误
   */
  dberr_t recover_from_aux(trx_t* trx, dict_index_t* index);

  /**
   * 预留 k 个 ID，返回起始 ID（线程安全）。
   * 调用者应在提交阶段调用，用于 add_with_ids。
   *
   * @param k      需要预留的数量
   * @param start  输出本次分配的起始 ID
   * @return DB_SUCCESS / DB_ERROR（溢出或未初始化且恢复失败）
   */
  dberr_t reserve(uint64_t k, uint64_t* start);

  /**
   * （可选）窥视下一号（仅调试/日志）。
   */
  uint64_t peek_next_unsafe() const noexcept { return next; }
};

/** —— 需要你实现/打通的钩子 —— 
 * 从该索引的辅助表读取 MAX(faiss_id)，写入 *out_max。
 * 若表为空，则返回 0（把 *out_max 设 0）。
 * 需要保证在快照一致或可重复读下读取，避免并发抖动。
 *
 * @return DB_SUCCESS / 其他错误
 */
dberr_t vec_aux_select_max_id(trx_t* trx, dict_index_t* index, uint64_t* out_max, bool* empty);
