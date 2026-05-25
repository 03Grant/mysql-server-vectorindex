// vec_merger.cc
#include "vec_merger.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "current_thd.h"
#include "data0type.h"
#include "dict0dict.h"
#include "dict0mem.h"
#include "ha_innodb.h"
#include "lob0lob.h"
#include "mtr0mtr.h"
#include "mem0mem.h"
#include "my_sys.h"
#include "my_thread.h"
#include "os0file.h"
#include "rem0rec.h"
#include "row0row.h"
#include "row0sel.h"
#include "sql/sql_class.h"
#include "sql/sql_lex.h"
#include "sql/sql_thd_internal_api.h"
#include "trx0trx.h"
#include "ut0ut.h"

#include "vec_aux_tables.h"
#include "vec_faiss_factory.h"
#include "vec_hnswlib_factory.h"
#include "vec_diskann_factory.h"
#include "vec_index_runtime.h"
#include "vec_meta.h"

namespace {

constexpr size_t kVecMergeMinSegments = 2;

static bool vec_should_skip_task(const dict_index_t* index) {
  if (index == nullptr || index->table == nullptr) {
    return true;
  }
  if (index->table->to_be_dropped) {
    return true;
  }
  return vec_dict_table_is_aux(index->table);
}

class VecBgThdGuard {
 public:
  VecBgThdGuard() = default;
  ~VecBgThdGuard() { cleanup(); }

  bool create() {
    prev_thd_ = current_thd;
    thd_ = create_thd(false, true, true, 0, 0);
    if (thd_ == nullptr) {
      current_thd = prev_thd_;
      return false;
    }
    current_thd = thd_;
    thd_->set_command(COM_DAEMON);
    thd_->variables.option_bits &= ~OPTION_AUTOCOMMIT;
    thd_->variables.option_bits |= OPTION_NOT_AUTOCOMMIT;
    thd_->variables.transaction_read_only = false;
    thd_->tx_read_only = false;
    thd_->system_thread = SYSTEM_THREAD_BACKGROUND;
    thd_->security_context()->skip_grants();
    thd_->lex->sql_command = SQLCOM_CREATE_TABLE;
    return true;
  }

  bool start_trx() {
    if (thd_ == nullptr) {
      return false;
    }
    trx_t *&trx_ref = thd_to_trx(thd_);
    if (trx_ref == nullptr) {
      trx_ref = innobase_trx_allocate(thd_);
      owns_trx_ = true;
    } else {
      owns_trx_ = false;
    }
    trx_ = trx_ref;
    if (trx_ == nullptr) {
      return false;
    }
    trx_start_if_not_started(trx_, true, UT_LOCATION_HERE);
    return true;
  }

  trx_t *trx() const { return trx_; }
  THD *thd() const { return thd_; }

 private:
  void cleanup() {
    if (thd_ != nullptr) {
      current_thd = thd_;
      if (trx_t *trx = thd_to_trx(thd_); trx != nullptr) {
        (void)trx_rollback_for_mysql(trx);
      }
      destroy_thd(thd_);
      thd_ = nullptr;
      current_thd = prev_thd_;
    }
    trx_ = nullptr;
    owns_trx_ = false;
  }

  THD *thd_{nullptr};
  trx_t *trx_{nullptr};
  THD *prev_thd_{nullptr};
  bool owns_trx_{false};
};

inline bool vec_cache_read_u32(const unsigned char *&p, size_t &remain,
                               uint32_t &out) {
  if (remain < sizeof(uint32_t)) {
    return false;
  }
  out = static_cast<uint32_t>(p[0]) |
        (static_cast<uint32_t>(p[1]) << 8) |
        (static_cast<uint32_t>(p[2]) << 16) |
        (static_cast<uint32_t>(p[3]) << 24);
  p += sizeof(uint32_t);
  remain -= sizeof(uint32_t);
  return true;
}

class VecBaseRowReader {
 public:
  enum class Status { kOk, kNotFound, kInvalidVec, kError };

  VecBaseRowReader(dict_index_t *clust_index,
                   ulint vec_field_no,
                   size_t vec_dim,
                   trx_t *trx)
      : clust_index_(clust_index),
        vec_field_no_(vec_field_no),
        vec_dim_(vec_dim),
        trx_(trx) {
    if (clust_index_ == nullptr || vec_field_no_ == ULINT_UNDEFINED ||
        vec_dim_ == 0 || trx_ == nullptr) {
      return;
    }
    total_fields_ = clust_index_->n_fields;
    pk_fields_ = dict_index_get_n_unique(clust_index_);
    if (pk_fields_ == 0 || total_fields_ == 0) {
      return;
    }
    heap_ = mem_heap_create(256, UT_LOCATION_HERE);
    if (heap_ == nullptr) {
      return;
    }
    tuple_ = dtuple_create(heap_, pk_fields_);
    if (tuple_ == nullptr) {
      return;
    }
    dict_index_copy_types(tuple_, clust_index_, pk_fields_);
    ready_ = true;
  }

  ~VecBaseRowReader() {
    if (heap_ != nullptr) {
      mem_heap_free(heap_);
      heap_ = nullptr;
    }
  }

  bool ok() const { return ready_; }

  Status read(const unsigned char *pk_entry,
              size_t pk_len,
              trx_id_t *out_trx_id,
              std::vector<float> &out_vec) {
    if (!ready_ || pk_entry == nullptr || pk_len == 0 || tuple_ == nullptr) {
      return Status::kError;
    }

    if (!bind_tuple(pk_entry, pk_len)) {
      return Status::kError;
    }

    mtr_t mtr;
    mtr_start(&mtr);
    btr_pcur_t pcur;
    const bool found = row_search_on_row_ref(
        &pcur, BTR_SEARCH_LEAF, clust_index_->table, tuple_, &mtr);
    if (!found) {
      pcur.close();
      mtr_commit(&mtr);
      return Status::kNotFound;
    }

    const rec_t *rec = pcur.get_rec();
    if (rec == nullptr || !page_rec_is_user_rec(rec) ||
        rec_get_deleted_flag(rec, dict_table_is_comp(clust_index_->table))) {
      pcur.close();
      mtr_commit(&mtr);
      return Status::kNotFound;
    }

    Rec_offsets offsets_holder;
    const ulint *offsets = offsets_holder.compute(rec, clust_index_);

    if (out_trx_id != nullptr) {
      *out_trx_id = row_get_rec_trx_id(rec, clust_index_, offsets);
    }

    const byte *data = nullptr;
    ulint len = 0;
    mem_heap_t *ext_heap = nullptr;

    if (rec_offs_nth_extern(clust_index_, offsets, vec_field_no_)) {
      ext_heap = mem_heap_create(1, UT_LOCATION_HERE);
      data = lob::btr_rec_copy_externally_stored_field(
          trx_, clust_index_, rec, offsets,
          dict_table_page_size(clust_index_->table), vec_field_no_, &len,
          nullptr, dict_index_is_sdi(clust_index_), ext_heap);
    } else {
      data = rec_get_nth_field_instant(rec, offsets, vec_field_no_,
                                       clust_index_, &len);
    }

    bool vec_ok = true;
    const size_t expected = vec_dim_ * sizeof(float);
    if (data == nullptr || len == UNIV_SQL_NULL || len != expected) {
      vec_ok = false;
    } else {
      out_vec.resize(vec_dim_);
      for (size_t i = 0; i < vec_dim_; ++i) {
        float value;
        std::memcpy(&value, data + i * sizeof(float), sizeof(float));
        if (!std::isfinite(value)) {
          vec_ok = false;
          break;
        }
        out_vec[i] = value;
      }
    }

    if (ext_heap != nullptr) {
      mem_heap_free(ext_heap);
    }

    pcur.close();
    mtr_commit(&mtr);

    return vec_ok ? Status::kOk : Status::kInvalidVec;
  }

 private:
  bool bind_tuple(const unsigned char *data, size_t len) {
    if (data == nullptr || tuple_ == nullptr) {
      return false;
    }
    const unsigned char *p = data;
    size_t remain = len;

    uint32_t num_cols = 0;
    if (!vec_cache_read_u32(p, remain, num_cols)) {
      return false;
    }
    if (num_cols != pk_fields_) {
      return false;
    }

    dtuple_set_n_fields(tuple_, pk_fields_);
    dtuple_set_n_fields_cmp(tuple_, pk_fields_);

    for (uint32_t i = 0; i < num_cols; ++i) {
      uint32_t len_field = 0;
      if (!vec_cache_read_u32(p, remain, len_field)) {
        return false;
      }
      dfield_t *df = dtuple_get_nth_field(tuple_, i);
      if (len_field == std::numeric_limits<uint32_t>::max()) {
        dfield_set_null(df);
        continue;
      }
      if (remain < len_field) {
        return false;
      }
      dfield_set_data(df, const_cast<unsigned char *>(p), len_field);
      p += len_field;
      remain -= len_field;
    }

    return true;
  }

  dict_index_t *clust_index_{nullptr};
  ulint vec_field_no_{ULINT_UNDEFINED};
  size_t vec_dim_{0};
  trx_t *trx_{nullptr};
  mem_heap_t *heap_{nullptr};
  dtuple_t *tuple_{nullptr};
  ulint pk_fields_{0};
  ulint total_fields_{0};
  bool ready_{false};
};

static bool vec_pick_merge_segments(dict_index_t *index,
                                    std::string *out_s1,
                                    std::string *out_s2,
                                    size_t *out_count = nullptr) {
  if (index == nullptr || index->vec_runtime == nullptr ||
      out_s1 == nullptr || out_s2 == nullptr) {
    return false;
  }
  vec_index_ctx_t *ctx = index->vec_runtime;

  std::vector<std::pair<uint32_t, std::string>> candidates;
  {
    std::shared_lock<std::shared_mutex> lk(ctx->mu);
    for (const auto &seg : ctx->segments) {
      if (!seg.immutable) {
        continue;
      }
      const uint32_t seg_id = vec_segment_id_to_u32(seg.vecindex_id);
      if (seg_id == 0) {
        continue;
      }
      if (seg_id == ctx->max_vecindex_id) {
        continue;
      }
      candidates.push_back({seg_id, seg.vecindex_id});
    }
  }

  if (out_count != nullptr) {
    *out_count = candidates.size();
  }

  if (candidates.size() < 2) {
    return false;
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });

  *out_s1 = candidates[0].second;
  *out_s2 = candidates[1].second;
  return true;
}

static bool vec_load_segment_mapping(vec_index_segment_t *seg,
                                     vid_pk_mapping_t *out) {
  if (seg == nullptr || out == nullptr || seg->rw_lock == nullptr) {
    return false;
  }

  {
    std::shared_lock<std::shared_mutex> seg_lock(*seg->rw_lock);
    if (seg->vid_pk_mapping.ready) {
      *out = seg->vid_pk_mapping;
      return true;
    }
  }

  // if (seg->index_file_name.empty()) {
  //   return false;
  // }
  // const std::string path = vec_vid_pk_mapping_path(seg->index_file_name);
  // if (path.empty()) {
  //   return false;
  // }

  // vid_pk_mapping_t loaded;
  // if (!vec_vid_pk_mapping_load(path, &loaded)) {
  //   return false;
  // }
  // loaded.ready = true;

  // {
  //   std::unique_lock<std::shared_mutex> seg_lock(*seg->rw_lock);
  //   seg->vid_pk_mapping = loaded;
  // }
  // *out = std::move(loaded);
  ib::warn() << "VECMERGE: vec_load_segment_mapping is not ready yet.";
  return false;
}

static bool vec_resolve_merge_path(const dict_index_t *index,
                                   const std::string &s1,
                                   const std::string &s2,
                                   std::string *out) {
  if (index == nullptr || out == nullptr || s1.empty() || s2.empty()) {
    return false;
  }
  const uint32_t s1_id = vec_segment_id_to_u32(s1);
  const uint32_t s2_id = vec_segment_id_to_u32(s2);
  if (s1_id == 0 || s2_id == 0) {
    return false;
  }
  std::string meta_path;
  if (!vec_meta_path_for_index(index, &meta_path)) {
    return false;
  }
  const std::string dir = vec_meta_dirname(meta_path);
  if (dir.empty()) {
    return false;
  }
  std::string file = "vecseg_";
  file.append(std::to_string(
      static_cast<unsigned long long>(index->table->id)));
  file.push_back('_');
  file.append(std::to_string(
      static_cast<unsigned long long>(index->id)));
  file.push_back('_');
  file.append(s1);
  file.push_back('\'');
  file.append(s2);
  file.append(".vec");
  *out = vec_meta_join(dir, file);
  return true;
}

static void vec_cleanup_segment_files(const std::string &seg_path) {
  if (seg_path.empty()) {
    return;
  }
  auto drop_one = [](const std::string &path) {
    if (path.empty()) {
      return;
    }
    bool existed = false;
    if (!os_file_delete_if_exists(innodb_data_file_key, path.c_str(), &existed) &&
        existed) {
      ib::warn() << "VECMERGE: failed to delete file '" << path << "'";
    }
  };

  std::vector<std::string> segment_files;
  vec_diskann_collect_artifact_paths(seg_path, &segment_files);
  if (segment_files.empty()) {
    segment_files.push_back(seg_path);
  }
  for (const auto &path : segment_files) {
    drop_one(path);
  }
  const std::string pkmap_path = vec_vid_pk_mapping_path(seg_path);
  drop_one(pkmap_path);
}

}  // namespace

VecMergeManager &VecMergeManager::instance() {
  static VecMergeManager instance;
  return instance;
}

VecMergeManager::~VecMergeManager() { stop(); }

void VecMergeManager::ensure_started() {
  if (!running.load(std::memory_order_acquire)) {
    start();
  }
}

void VecMergeManager::start() {
  bool expected = false;
  if (!running.compare_exchange_strong(expected, true,
                                       std::memory_order_acq_rel)) {
    return;
  }
  worker_thread = std::thread(&VecMergeManager::worker_loop, this);
}

void VecMergeManager::stop() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    if (!running.load(std::memory_order_acquire)) {
      return;
    }
    running.store(false, std::memory_order_release);
  }
  cv.notify_all();
  if (worker_thread.joinable()) {
    worker_thread.join();
  }
}

void VecMergeManager::submit_task(dict_index_t *index) {
  if (index == nullptr || index->vec_runtime == nullptr) {
    return;
  }
  std::string s1;
  std::string s2;
  if (!vec_pick_merge_segments(index, &s1, &s2)) {
    return;
  }
  submit_task(index, s1, s2);
}

void VecMergeManager::submit_task(dict_index_t *index,
                                  const std::string &seg_left,
                                  const std::string &seg_right) {
  if (index == nullptr || seg_left.empty() || seg_right.empty()) {
    return;
  }
  if (merge_busy.exchange(true)) {
    return;
  }
  ensure_started();
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    tasks.push({index, seg_left, seg_right});
  }
  ib::warn() << "VECMERGE: task submitted for index id="
             << static_cast<unsigned long long>(index->id)
             << " segs=" << seg_left << "," << seg_right;
  cv.notify_one();
}

void VecMergeManager::worker_loop() {
  bool thread_initialized = !my_thread_init();
  struct ThreadGuard {
    bool ok{false};
    ~ThreadGuard() {
      if (ok) {
        my_thread_end();
      }
    }
  } thread_guard{thread_initialized};

  if (!thread_initialized) {
    running.store(false, std::memory_order_release);
    cv.notify_all();
    return;
  }

  while (true) {
    VecMergeTask task{};
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      cv.wait(lock, [this] {
        return !tasks.empty() || !running.load(std::memory_order_acquire);
      });

      if (!running.load(std::memory_order_acquire) && tasks.empty()) {
        return;
      }

      task = tasks.front();
      tasks.pop();
    }
    process_task(task);
  }
}

void VecMergeManager::process_task(VecMergeTask task) {
  struct BusyGuard {
    std::atomic<bool> *flag{nullptr};
    ~BusyGuard() {
      if (flag != nullptr) {
        flag->store(false, std::memory_order_release);
      }
    }
  } busy_guard{&merge_busy};

  dict_index_t *index = task.index;
  if (index == nullptr || index->table == nullptr ||
      index->vec_runtime == nullptr) {
    return;
  }
  if (vec_should_skip_task(index)) {
    return;
  }

  vec_index_ctx_t *ctx = index->vec_runtime;
  const std::string seg1_id = task.seg_left;
  const std::string seg2_id = task.seg_right;

  vec_index_segment_t *seg1 = nullptr;
  vec_index_segment_t *seg2 = nullptr;
  std::string seg1_path;
  std::string seg2_path;

  {
    std::shared_lock<std::shared_mutex> ctx_lock(ctx->mu);
    seg1 = vec_find_segment_by_id(ctx, seg1_id);
    seg2 = vec_find_segment_by_id(ctx, seg2_id);
    if (seg1 == nullptr || seg2 == nullptr ||
        !seg1->immutable || !seg2->immutable) {
      return;
    }
    seg1_path = seg1->index_file_name;
    seg2_path = seg2->index_file_name;
  }

  if (seg1_path.empty() || seg2_path.empty()) {
    ib::warn() << "VECMERGE: missing segment file path, skip merge segs="
               << seg1_id << "," << seg2_id;
    return;
  }

  auto pkmap_exists = [](const std::string &seg_path) -> bool {
    const std::string pkmap = vec_vid_pk_mapping_path(seg_path);
    if (pkmap.empty()) {
      return false;
    }
    return my_access(pkmap.c_str(), F_OK) == 0;
  };
  if (!pkmap_exists(seg1_path) || !pkmap_exists(seg2_path)) {
    ib::warn() << "VECMERGE: pkmap file missing, skip merge segs="
               << seg1_id << "," << seg2_id;
    return;
  }

  vid_pk_mapping_t map1;
  vid_pk_mapping_t map2;
  if (!vec_load_segment_mapping(seg1, &map1) ||
      !vec_load_segment_mapping(seg2, &map2)) {
    ib::warn() << "VECMERGE: pk mapping missing, skip merge segs="
               << seg1_id << "," << seg2_id;
    return;
  }

  if (map1.key_length != 0 && map2.key_length != 0 &&
      map1.key_length != map2.key_length) {
    ib::warn() << "VECMERGE: pk mapping key length mismatch, segs="
               << seg1_id << "," << seg2_id;
    return;
  }

  const size_t dim = ctx->params.dim;
  if (dim == 0) {
    ib::warn() << "VECMERGE: invalid dim for index id="
               << static_cast<unsigned long long>(index->id);
    return;
  }

  dict_index_t *clust_index = index->table->first_index();
  if (clust_index == nullptr) {
    return;
  }

  const dict_field_t *vec_field = index->get_field(0);
  if (vec_field == nullptr || vec_field->col == nullptr) {
    return;
  }
  const size_t vec_col_no = dict_col_get_no(vec_field->col);
  const ulint vec_field_no =
      dict_table_get_nth_col_pos(clust_index->table, vec_col_no);
  if (vec_field_no == ULINT_UNDEFINED) {
    return;
  }

  VecBgThdGuard thd_guard;
  if (!thd_guard.create() || !thd_guard.start_trx()) {
    ib::warn() << "VECMERGE: failed to create THD/trx for merge";
    return;
  }

  VecBaseRowReader row_reader(clust_index, vec_field_no, dim,
                              thd_guard.trx());
  if (!row_reader.ok()) {
    ib::warn() << "VECMERGE: failed to init row reader";
    return;
  }

  vid_pk_mapping_t merged_mapping;
  merged_mapping.key_length = (map1.key_length != 0) ? map1.key_length
                                                     : map2.key_length;
  merged_mapping.ready = true;

  std::vector<float> xb;
  std::vector<int64_t> ids;

  auto consume_mapping = [&](const vid_pk_mapping_t &mapping) {
    const size_t total = mapping.pk_values.size();
    for (size_t i = 0; i < total; ++i) {
      const auto &entry = mapping.pk_values[i];
      if (entry.empty()) {
        continue;
      }
      const trx_id_t map_trx =
          (i < mapping.trx_ids.size()) ? mapping.trx_ids[i] : 0;
      if (map_trx == 0) {
        continue;
      }
      trx_id_t row_trx = 0;
      std::vector<float> vec_values;
      const auto status =
          row_reader.read(entry.data(), entry.size(), &row_trx, vec_values);
      if (status != VecBaseRowReader::Status::kOk) {
        continue;
      }
      if (row_trx != map_trx) {
        continue;
      }
      if (merged_mapping.key_length == 0) {
        merged_mapping.key_length = entry.size();
      }
      const uint64_t new_id = static_cast<uint64_t>(ids.size());
      ids.push_back(static_cast<int64_t>(new_id));
      xb.insert(xb.end(), vec_values.begin(), vec_values.end());
      merged_mapping.pk_values.push_back(entry);
      merged_mapping.trx_ids.push_back(map_trx);
    }
  };

  consume_mapping(map1);
  consume_mapping(map2);

  std::unique_ptr<IVectorIndex> target;
  switch (ctx->params.backend) {
    case BackendType::Faiss:
      target = vec_make_faiss_index(ctx->params);
      break;
    case BackendType::Hnswlib:
      target = vec_make_hnswlib_index(ctx->params);
      break;
    case BackendType::Diskann:
      target = vec_make_diskann_index(ctx->params);
      break;
    default:
      break;
  }
  if (!target) {
    ib::warn() << "VECMERGE: failed to create target index";
    return;
  }

  if (!ids.empty()) {
    target->train(ids.size(), xb.data());
    target->add(ids.size(), xb.data(), ids.data());
  }

  std::string merge_path;
  if (!vec_resolve_merge_path(index, seg1_id, seg2_id, &merge_path)) {
    ib::warn() << "VECMERGE: failed to resolve merge path for segs="
               << seg1_id << "," << seg2_id;
    return;
  }

  const std::string merge_basename = vec_meta_basename(merge_path);
  const uint64_t seg1_num = static_cast<uint64_t>(vec_segment_id_to_u32(seg1_id));

  if (!vec_meta_append_event(index, ctx->params, seg1_num,
                             static_cast<uint64_t>(ids.size()),
                             VecSegmentState::Preparing,
                             merge_basename, false)) {
    ib::warn() << "VECMERGE: failed to append Preparing event";
  }

  target->save(merge_path);
  if (ctx->params.backend == BackendType::Diskann) {
    target->load(merge_path);
  }

  if (!vec_meta_append_event(index, ctx->params, seg1_num,
                             static_cast<uint64_t>(ids.size()),
                             VecSegmentState::BuiltIndex,
                             merge_basename, false)) {
    ib::warn() << "VECMERGE: failed to append BuiltIndex event";
  }

  const std::string merge_pkmap = vec_vid_pk_mapping_path(merge_path);
  bool pkmap_saved = false;
  if (!merge_pkmap.empty()) {
    pkmap_saved = vec_vid_pk_mapping_save(merged_mapping, merge_pkmap);
    if (!pkmap_saved) {
      ib::warn() << "VECMERGE: failed to save pk mapping to '"
                 << merge_pkmap << "'";
      vec_cleanup_segment_files(merge_path);
      return;
    }
    if (!vec_meta_append_event(index, ctx->params, seg1_num,
                               static_cast<uint64_t>(ids.size()),
                               VecSegmentState::PkmapSaved,
                               merge_basename, true)) {
      ib::warn() << "VECMERGE: failed to append PkmapSaved event";
    }
  }

  vec_index_segment_t merged{};
  merged.index = std::move(target);
  merged.immutable = true;
  merged.vecindex_id = seg1_id;
  merged.index_file_name = merge_path;
  merged.vid_pk_mapping = std::move(merged_mapping);
  merged.vid_pk_mapping.ready = true;

  bool swapped = false;
  {
    std::unique_lock<std::shared_mutex> ctx_lock(ctx->mu);
    vec_index_segment_t *cur1 = vec_find_segment_by_id(ctx, seg1_id);
    vec_index_segment_t *cur2 = vec_find_segment_by_id(ctx, seg2_id);
    if (cur1 == nullptr || cur2 == nullptr ||
        !cur1->immutable || !cur2->immutable) {
      vec_cleanup_segment_files(merge_path);
      return;
    }

    std::shared_ptr<std::shared_mutex> rw1 = cur1->rw_lock;
    std::shared_ptr<std::shared_mutex> rw2 = cur2->rw_lock;
    std::unique_lock<std::shared_mutex> lk1;
    std::unique_lock<std::shared_mutex> lk2;
    if (rw1) {
      lk1 = std::unique_lock<std::shared_mutex>(*rw1);
    }
    if (rw2) {
      lk2 = std::unique_lock<std::shared_mutex>(*rw2);
    }

    auto it1 = ctx->segment_id_map.find(seg1_id);
    auto it2 = ctx->segment_id_map.find(seg2_id);
    if (it1 == ctx->segment_id_map.end() ||
        it2 == ctx->segment_id_map.end() ||
        it1->second == it2->second) {
      vec_cleanup_segment_files(merge_path);
      return;
    }
    size_t idx1 = it1->second;
    size_t idx2 = it2->second;
    size_t hi = std::max(idx1, idx2);
    size_t lo = std::min(idx1, idx2);
    if (hi < ctx->segments.size()) {
      ctx->segments.erase(ctx->segments.begin() + hi);
    }
    if (lo < ctx->segments.size()) {
      ctx->segments.erase(ctx->segments.begin() + lo);
    }
    ctx->segments.push_back(std::move(merged));
    vec_rebuild_segment_id_map(ctx);
    swapped = true;
  }

  if (!swapped) {
    vec_cleanup_segment_files(merge_path);
    return;
  }

  if (!vec_meta_append_event(index, ctx->params, seg1_num,
                             static_cast<uint64_t>(ids.size()),
                             VecSegmentState::Committed,
                             merge_basename, pkmap_saved)) {
    ib::warn() << "VECMERGE: failed to append Committed event";
  }

  const uint64_t seg2_num = static_cast<uint64_t>(vec_segment_id_to_u32(seg2_id));
  if (seg2_num != 0) {
    if (!vec_meta_append_event(index, ctx->params, seg2_num, 0,
                               VecSegmentState::Tombstone,
                               "", false)) {
      ib::warn() << "VECMERGE: failed to append Tombstone for seg_id="
                 << seg2_id;
    }
  }

  vec_cleanup_segment_files(seg1_path);
  vec_cleanup_segment_files(seg2_path);

  ib::warn() << "VECMERGE: merge completed segs=" << seg1_id << ","
             << seg2_id << " new_count=" << ids.size();
}

bool vec_schedule_merge_if_needed(dict_index_t *index) {
  if (index == nullptr || index->vec_runtime == nullptr) {
    return false;
  }
  size_t immutable_count = 0;
  std::string s1;
  std::string s2;
  if (!vec_pick_merge_segments(index, &s1, &s2, &immutable_count)) {
    return false;
  }
  if (immutable_count < kVecMergeMinSegments) {
    return false;
  }
  VecMergeManager::instance().submit_task(index, s1, s2);
  return true;
}
