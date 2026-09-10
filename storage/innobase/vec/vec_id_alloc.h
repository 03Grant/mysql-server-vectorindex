#pragma once

#include <cstdint>
#include <mutex>
#include <limits>
#include "db0err.h"
#include "trx0trx.h"


// Forward declarations from InnoDB
struct trx_t;
struct dict_index_t;

/**
 * Monotonic ID allocator for a vector index, one per dict_index_t.
 * - Lazy recovery: read MAX(faiss_id) from the auxiliary table on the first recover_from_aux() call.
 * - Allocation: reserve(k, &start) reserves the range [start, start+k-1].
 */
struct vec_id_allocator_t {
  // Runtime state
  std::mutex mu{};
  uint64_t   next{0};
  bool       inited{false};

  // === Lifecycle ===

  /**
   * Lazy recovery: read MAX(faiss_id) from the auxiliary table and set next to max+1.
   * May be called repeatedly; only takes effect when not initialized.
   *
   * @param trx   Transaction handle for reading the auxiliary table
   * @param index Vector index used to locate the auxiliary table
   * @return DB_SUCCESS or another error code
   */
  dberr_t recover_from_aux(trx_t* trx, dict_index_t* index);

  /**
   * Reserve k IDs and return the starting ID (thread-safe).
   * Call during commit for use with add_with_ids.
   *
   * @param k      Number of IDs to reserve
   * @param start  Output: starting ID of this allocation
   * @return DB_SUCCESS / DB_ERROR (overflow, or uninitialized with failed recovery)
   */
  dberr_t reserve(uint64_t k, uint64_t* start);

  /**
   * Optionally peek at the next ID for debugging/logging only.
   */
  uint64_t peek_next_unsafe() const noexcept { return next; }
};

/** --- Hooks to implement or integrate ---
 * Read MAX(faiss_id) from the index's auxiliary table into *out_max.
 * For an empty table, report a maximum of 0 by setting *out_max to 0.
 * Use a consistent snapshot or repeatable read to avoid inconsistent concurrent reads.
 *
 * @return DB_SUCCESS or another error code
 */
dberr_t vec_aux_select_max_id(trx_t* trx, dict_index_t* index, uint64_t* out_max, bool* empty);
