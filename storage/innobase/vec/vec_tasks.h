#pragma once

#include <atomic>
#include <cstdint>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>

#include "db0err.h"
#include "dict0types.h"

struct dict_index_t;
struct dict_table_t;
struct trx_t;
struct vec_index_ctx_t;
struct vec_index_segment_t;
struct vid_pk_mapping_t;
class THD;

struct VecBuildTask {
  dict_index_t* index{nullptr};
  space_index_t index_id{0};
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

/* Performs the lightweight rotation (swap mutable/immutable segments).
Caller must provide an active transaction. Returns true on success. */
bool vec_rotate_mem_index(trx_t* trx, dict_index_t* index);

// SINGLEAXU:DELETE
/* Load vid->PK mapping for a specific immutable segment. Prefer persisted
   mapping files when available. */
bool vec_load_aux_cache_for_segment(dict_index_t* vec_index,
                                    vec_index_ctx_t* ctx,
                                    vec_index_segment_t* seg, THD* thd);

/* Schedule a background bootstrap load from vec metadata on startup. */
void vec_schedule_bootstrap_load(dict_index_t* index);

/* Block until no rotation/flush task is pending or running for any vector
index of the given table. Called at the end of the DDL ingestion scan so the
initial segment build reaches its Committed manifest state before the DDL
statement returns: a crash after ADD VECINDEX completes can then never lose
the base segment, and a crash before this point aborts the DDL entirely. */
void vec_wait_table_builds_idle(dict_table_t* table);

/* Install a native DiskANN immutable segment built directly from streamed DDL
   rows, then create a fresh empty mutable delta segment. */
dberr_t vec_complete_direct_diskann_build(dict_index_t* index,
                                          const std::string& data_path,
                                          uint64_t row_count,
                                          const vid_pk_mapping_t& mapping);
