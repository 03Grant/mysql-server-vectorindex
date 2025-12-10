#pragma once

#include <atomic>
#include <cstdint>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <set>
#include <thread>

#include "dict0types.h"

struct dict_index_t;
struct trx_t;
struct vec_index_ctx_t;
struct vec_index_segment_t;
class THD;

struct VecBuildTask {
  dict_index_t* index{nullptr};
};

class VecTaskManager {
 public:
  static VecTaskManager& instance();

  void submit_task(dict_index_t* index);

  void start();
  void stop();

  /* Rotate mutable segment -> immutable + new mutable. */
  friend bool vec_rotate_mem_index(trx_t* trx, dict_index_t* index);

 private:
  VecTaskManager() = default;
  ~VecTaskManager();

  VecTaskManager(const VecTaskManager&) = delete;
  VecTaskManager& operator=(const VecTaskManager&) = delete;

  void ensure_started();
  void worker_loop();
  void process_task(VecBuildTask task);

  std::queue<VecBuildTask> tasks;
  std::set<uint64_t> pending_index_ids;
  std::mutex queue_mutex;
  std::condition_variable cv;
  std::thread worker_thread;
  std::atomic<bool> running{false};
};

/* Performs the lightweight rotation (rename current mutable aux table to a
segment, create fresh mutable aux table, swap runtime segments).
Caller must provide an active transaction. Returns true on success. */
bool vec_rotate_mem_index(trx_t* trx, dict_index_t* index);

/* Prepare a fresh aux table in the caller (user) THD before scheduling a
rotation. This creates the table and registers DD immediately, leaving it
ready to be renamed to *_MEM by the background rotate task. */
bool vec_prepare_pending_mem_table(dict_index_t* index);

/* Load vid->PK mapping for a specific immutable segment. Prefer persisted
   mapping files when available; fall back to scanning the aux table. */
bool vec_load_aux_cache_for_segment(dict_index_t* vec_index,
                                    vec_index_ctx_t* ctx,
                                    vec_index_segment_t* seg, THD* thd);

/* Schedule a background bootstrap load from vec metadata on startup. */
void vec_schedule_bootstrap_load(dict_index_t* index);
