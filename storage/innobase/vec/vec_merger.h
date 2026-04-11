#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

struct dict_index_t;

struct VecMergeTask {
  dict_index_t* index{nullptr};
  std::string seg_left;
  std::string seg_right;
};

class VecMergeManager {
 public:
  static VecMergeManager& instance();

  // Pick oldest two immutable segments and submit a merge task.
  void submit_task(dict_index_t* index);

  // Submit a merge task with explicit segment ids.
  void submit_task(dict_index_t* index,
                   const std::string& seg_left,
                   const std::string& seg_right);

  void start();
  void stop();

 private:
  VecMergeManager() = default;
  ~VecMergeManager();

  VecMergeManager(const VecMergeManager&) = delete;
  VecMergeManager& operator=(const VecMergeManager&) = delete;

  void ensure_started();
  void worker_loop();
  void process_task(VecMergeTask task);

  std::mutex queue_mutex;
  std::condition_variable cv;
  std::queue<VecMergeTask> tasks;
  std::thread worker_thread;
  std::atomic<bool> running{false};
  std::atomic<bool> merge_busy{false};
};

// Schedule a merge task if eligible (enforces global single-merge rule).
bool vec_schedule_merge_if_needed(dict_index_t* index);
