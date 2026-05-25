#include "vec_diskann_factory.h"

#include <immintrin.h>
#include "disk_utils.h"
#include "distance.h"
#include "index.h"
#include "parameters.h"
#include "pq_flash_index.h"
#include "utils.h"
#include "ut0ut.h"

#ifndef _WINDOWS
#include "linux_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using DiskannIndexT = diskann::Index<float, uint32_t, uint32_t>;
using DiskannPQFlashT = diskann::PQFlashIndex<float, uint32_t>;

constexpr uint32_t kDiskannTagOffset = 1;
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
// Keep MySQL's native DiskANN immutable search defaults aligned with the
// standalone search_disk_index benchmark unless explicitly overridden later.
constexpr uint32_t kDefaultBeamWidth = 2;
constexpr uint64_t kDefaultCacheNodes = 0;

constexpr std::array<const char *, 13> kDiskannArtifactSuffixes = {
    "_disk.index",
    "_disk.index_centroids.bin",
    "_disk.index_medoids.bin",
    "_disk.index_labels.txt",
    "_disk.index_labels_to_medoids.txt",
    "_disk.index_universal_label.txt",
    "_disk.index_labels_map.txt",
    "_disk.index_dummy_map.txt",
    "_disk.index_max_base_norm.bin",
    "_disk.index_pq_pivots.bin",
    "_pq_pivots.bin",
    "_pq_compressed.bin",
    "_sample_data.bin"};

diskann::Metric vec_to_diskann_metric(uint8_t metric_tag) {
  switch (metric_tag) {
    case VEC_M_IP:
      return diskann::Metric::INNER_PRODUCT;
    case VEC_M_COSINE:
      return diskann::Metric::COSINE;
    case VEC_M_L2:
    default:
      return diskann::Metric::L2;
  }
}

bool diskann_metric_prefers_small(uint8_t metric_tag) {
  return !(metric_tag == VEC_M_IP || metric_tag == VEC_M_COSINE);
}

uint32_t choose_diskann_degree(const vec_params_t &p) {
  if (p.type_tag == VEC_T_HNSW && p.hnsw_m > 0) {
    return static_cast<uint32_t>(p.hnsw_m);
  }
  return 16;
}

uint32_t choose_diskann_search_l(const vec_params_t &p) {
  if (p.efConstruction > 0) {
    return static_cast<uint32_t>(std::max(p.efConstruction, 32));
  }
  return 64;
}

size_t choose_diskann_capacity(const vec_params_t &p) {
  if (p.size > 0) {
    return static_cast<size_t>(p.size);
  }
  return 1024;
}

uint32_t vid_to_tag(int64_t id) {
  if (id < 0 ||
      id >= static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::out_of_range("DiskANN vector id out of uint32 range");
  }
  const uint64_t tag = static_cast<uint64_t>(id) + kDiskannTagOffset;
  if (tag > std::numeric_limits<uint32_t>::max()) {
    throw std::out_of_range("DiskANN tag overflow");
  }
  return static_cast<uint32_t>(tag);
}

int64_t tag_to_vid(uint32_t tag) {
  if (tag < kDiskannTagOffset) {
    return -1;
  }
  return static_cast<int64_t>(tag - kDiskannTagOffset);
}

double round_up(double value, double step) {
  return std::ceil(value / step) * step;
}

double estimate_search_dram_budget_gb(size_t rows, uint32_t dim,
                                      uint32_t max_degree) {
  const double pq_gib = (static_cast<double>(rows) * 32.0) / kGiB;
  const double cache_gib =
      (static_cast<double>(kDefaultCacheNodes) *
       (4.0 * static_cast<double>(max_degree) + 4.0 * static_cast<double>(dim))) /
      kGiB;
  return std::max(1.0, round_up(pq_gib + cache_gib + 0.25, 0.5));
}

double estimate_build_dram_budget_gb(size_t rows, uint32_t dim) {
  const double raw_gib =
      (static_cast<double>(rows) * static_cast<double>(dim) * 4.0) / kGiB;
  return std::max(8.0, std::ceil(raw_gib + 16.0));
}

std::string build_native_diskann_params(const vec_params_t &params,
                                        size_t points_num) {
  std::ostringstream oss;
  oss << choose_diskann_degree(params) << ' ' << choose_diskann_search_l(params)
      << ' '
      << estimate_search_dram_budget_gb(points_num, params.dim,
                                        choose_diskann_degree(params))
      << ' ' << estimate_build_dram_budget_gb(points_num, params.dim) << ' '
      << (params.build_threads > 0 ? params.build_threads : 1) << ' ' << 0
      << ' ' << 0 << ' ' << 0 << ' ' << 0;
  return oss.str();
}

std::shared_ptr<AlignedFileReader> make_aligned_reader() {
#ifdef _WINDOWS
  return std::make_shared<WindowsAlignedFileReader>();
#else
  return std::make_shared<LinuxAlignedFileReader>();
#endif
}

void append_unique(std::vector<std::string> *out, const std::string &path) {
  if (out == nullptr || path.empty()) {
    return;
  }
  if (std::find(out->begin(), out->end(), path) == out->end()) {
    out->push_back(path);
  }
}

std::vector<std::string> collect_diskann_artifacts(const std::string &prefix) {
  std::vector<std::string> out;
  if (prefix.empty()) {
    return out;
  }

  if (file_exists(prefix)) {
    append_unique(&out, prefix);
  }
  for (const char *suffix : kDiskannArtifactSuffixes) {
    const std::string path = prefix + suffix;
    if (file_exists(path)) {
      append_unique(&out, path);
    }
  }

  const std::string sample_ids = prefix + "_sample_ids.bin";
  if (file_exists(sample_ids)) {
    append_unique(&out, sample_ids);
  }

  return out;
}

void remove_diskann_artifacts(const std::string &prefix) {
  for (const auto &path : collect_diskann_artifacts(prefix)) {
    std::remove(path.c_str());
  }
}

struct ScopedFileCleanup {
  explicit ScopedFileCleanup(std::string path_in) : path(std::move(path_in)) {}
  ~ScopedFileCleanup() {
    if (!path.empty()) {
      std::remove(path.c_str());
    }
  }

  std::string path;
};

class DiskannVectorIndex : public IVectorIndex {
 public:
  explicit DiskannVectorIndex(const vec_params_t &params)
      : params_(params),
        write_params_(diskann::IndexWriteParametersBuilder(
                          choose_diskann_search_l(params),
                          choose_diskann_degree(params))
                          .with_num_threads(params.build_threads > 0
                                                ? static_cast<uint32_t>(
                                                      params.build_threads)
                                                : 1)
                          .build()),
        search_params_(choose_diskann_search_l(params),
                       params.build_threads > 0
                           ? static_cast<uint32_t>(params.build_threads)
                           : 1),
        default_search_l_(search_params_.initial_search_list_size),
        capacity_(choose_diskann_capacity(params_)) {
    memory_index_ = make_memory_index(capacity_);
  }

  size_t dim() const override { return params_.dim; }

  void train(size_t, const float *) override {
    // DiskANN mutable graph and native disk build are both train-free here.
  }

  void add(size_t n, const float *xb, const int64_t *ids) override {
    ensure_mutable_mode();
    if (memory_index_ == nullptr || xb == nullptr || n == 0) {
      return;
    }

    ensure_capacity(n);
    if (size_ == 0) {
      bulk_build(n, xb, ids);
      return;
    }

    for (size_t i = 0; i < n; ++i) {
      const int64_t vid = ids != nullptr ? ids[i] : static_cast<int64_t>(size_);
      const uint32_t tag = vid_to_tag(vid);
      const int rc = memory_index_->insert_point(xb + i * params_.dim, tag);
      if (rc != 0) {
        throw std::runtime_error("DiskANN insert_point failed");
      }
      size_ = std::max<size_t>(size_, static_cast<size_t>(vid + 1));
    }
  }

  void search(size_t nq, const float *xq, size_t k, int64_t *out_ids,
              float *out_distances,
              const VecRuntimeSearchParams *params) const override {
    if (xq == nullptr || out_ids == nullptr || out_distances == nullptr) {
      return;
    }

    const bool prefer_small = diskann_metric_prefers_small(params_.metric_tag);
    const uint32_t search_l = choose_runtime_search_l(params, k);

    if (memory_index_ != nullptr) {
      for (size_t qi = 0; qi < nq; ++qi) {
        float *d_row = out_distances + qi * k;
        int64_t *i_row = out_ids + qi * k;

        std::fill_n(d_row, k,
                    prefer_small ? std::numeric_limits<float>::infinity()
                                 : -std::numeric_limits<float>::infinity());
        std::fill_n(i_row, k, int64_t(-1));

        if (size_ == 0) {
          continue;
        }

        const size_t k_use = std::min(k, size_);
        std::vector<uint32_t> tags(k_use);
        std::vector<float> distances(k_use);
        std::vector<float *> res_vectors;
        const size_t found = memory_index_->search_with_tags(
            xq + qi * params_.dim, k_use, search_l, tags.data(),
            distances.data(), res_vectors);

        for (size_t j = 0; j < found; ++j) {
          i_row[j] = tag_to_vid(tags[j]);
          d_row[j] = prefer_small ? distances[j] : (1.0f - distances[j]);
        }
      }
      return;
    }

    if (flash_index_ == nullptr) {
      return;
    }

    for (size_t qi = 0; qi < nq; ++qi) {
      float *d_row = out_distances + qi * k;
      int64_t *i_row = out_ids + qi * k;

      std::fill_n(d_row, k,
                  prefer_small ? std::numeric_limits<float>::infinity()
                               : -std::numeric_limits<float>::infinity());
      std::fill_n(i_row, k, int64_t(-1));

      if (size_ == 0) {
        continue;
      }

      const size_t k_use = std::min(k, size_);
      std::vector<uint64_t> ids64(k_use,
                                  std::numeric_limits<uint64_t>::max());
      std::vector<float> distances(k_use);
      flash_index_->cached_beam_search(
          xq + qi * params_.dim, k_use, search_l, ids64.data(),
          distances.data(), choose_runtime_beam_width(search_l));

      for (size_t j = 0; j < k_use; ++j) {
        if (ids64[j] == std::numeric_limits<uint64_t>::max() ||
            ids64[j] >= size_) {
          continue;
        }
        i_row[j] = static_cast<int64_t>(ids64[j]);
        d_row[j] = distances[j];
      }
    }
  }

  void save(const std::string &path) const override {
    if (path.empty() || size_ == 0) {
      return;
    }
    if (memory_index_ == nullptr) {
      throw std::runtime_error(
          "DiskANN native immutable segment cannot be re-saved from PQFlash");
    }

    remove_diskann_artifacts(path);

    ScopedFileCleanup data_file(path + "_mysql_build_input.fbin");
    write_native_dataset_file(data_file.path);

    const std::string build_params =
        build_native_diskann_params(params_, size_);
    const int rc = diskann::build_disk_index<float>(
        data_file.path.c_str(), path.c_str(), build_params.c_str(),
        vec_to_diskann_metric(params_.metric_tag), false);
    if (rc != 0) {
      remove_diskann_artifacts(path);
      throw std::runtime_error("DiskANN native build_disk_index failed");
    }
  }

  void load(const std::string &path) override {
    if (vec_diskann_has_native_artifacts(path)) {
      load_native(path);
      return;
    }

    const size_t frozen_pts = DiskannIndexT::get_graph_num_frozen_points(path);
    memory_index_ = make_memory_index(0, frozen_pts);
    memory_index_->load(path.c_str(),
                        params_.build_threads > 0
                            ? static_cast<uint32_t>(params_.build_threads)
                            : 1,
                        default_search_l_);
    flash_index_.reset();
    file_reader_.reset();
    size_ = memory_index_->get_num_points();
  }

  size_t ntotal() const override { return size_; }

  bool reconstruct(size_t id, float *out) const override {
    if (memory_index_ == nullptr || out == nullptr) {
      return false;
    }
    uint32_t tag;
    try {
      tag = vid_to_tag(static_cast<int64_t>(id));
    } catch (...) {
      return false;
    }
    return memory_index_->get_vector_by_tag(tag, out) == 0;
  }

  void set_search_params(const VecRuntimeSearchParams &params) override {
    if (params.ef_search.has_value() && params.ef_search.value() > 0) {
      default_search_l_ = static_cast<uint32_t>(
          std::max(params.ef_search.value(), static_cast<int>(1)));
    }
  }

 private:
  vec_params_t params_;
  diskann::IndexWriteParameters write_params_;
  diskann::IndexSearchParams search_params_;
  std::unique_ptr<DiskannIndexT> memory_index_;
  std::shared_ptr<AlignedFileReader> file_reader_;
  std::unique_ptr<DiskannPQFlashT> flash_index_;
  size_t size_{0};
  uint32_t default_search_l_{64};
  uint32_t default_beam_width_{kDefaultBeamWidth};
  size_t capacity_{0};

  std::unique_ptr<DiskannIndexT> make_memory_index(
      size_t capacity, size_t num_frozen_pts = 1) const {
    const size_t effective_capacity = capacity > 0 ? capacity : 0;
    return std::make_unique<DiskannIndexT>(
        vec_to_diskann_metric(params_.metric_tag), params_.dim,
        effective_capacity,
        std::make_shared<diskann::IndexWriteParameters>(write_params_),
        std::make_shared<diskann::IndexSearchParams>(search_params_),
        num_frozen_pts, true, true, false, false, 0, false, false);
  }

  void bulk_build(size_t n, const float *xb, const int64_t *ids) {
    std::vector<uint32_t> tags;
    tags.reserve(n);
    size_t max_vid = 0;

    for (size_t i = 0; i < n; ++i) {
      const int64_t vid = ids != nullptr ? ids[i] : static_cast<int64_t>(i);
      tags.push_back(vid_to_tag(vid));
      max_vid = std::max(max_vid, static_cast<size_t>(vid + 1));
    }

    memory_index_->build(xb, n, tags);
    size_ = max_vid;
  }

  void ensure_capacity(size_t incoming) {
    if (memory_index_ == nullptr) {
      return;
    }
    const size_t needed = size_ + incoming;
    if (needed <= capacity_) {
      return;
    }

    size_t new_capacity =
        capacity_ > 0 ? capacity_ : choose_diskann_capacity(params_);
    while (new_capacity < needed) {
      new_capacity = std::max(new_capacity * 2, needed);
    }

    auto rebuilt = make_memory_index(new_capacity);
    std::vector<float> buffer(params_.dim);
    for (size_t vid = 0; vid < size_; ++vid) {
      if (!reconstruct(vid, buffer.data())) {
        throw std::runtime_error(
            "DiskANN rebuild failed during capacity growth");
      }
      const uint32_t tag = vid_to_tag(static_cast<int64_t>(vid));
      if (rebuilt->insert_point(buffer.data(), tag) != 0) {
        throw std::runtime_error("DiskANN rebuild insert failed");
      }
    }

    memory_index_ = std::move(rebuilt);
    capacity_ = new_capacity;
  }

  uint32_t choose_runtime_search_l(const VecRuntimeSearchParams *params,
                                   size_t k) const {
    if (params != nullptr && params->ef_search.has_value() &&
        params->ef_search.value() > 0) {
      return static_cast<uint32_t>(
          std::max<size_t>(k, static_cast<size_t>(params->ef_search.value())));
    }
    return std::max<uint32_t>(default_search_l_, static_cast<uint32_t>(k));
  }

  uint32_t choose_runtime_beam_width(uint32_t search_l) const {
    return std::max<uint32_t>(
        1, std::min<uint32_t>(default_beam_width_, search_l));
  }

  void ensure_mutable_mode() const {
    if (memory_index_ == nullptr) {
      throw std::runtime_error(
          "DiskANN immutable native segment does not support add/build");
    }
  }

  void write_native_dataset_file(const std::string &path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
      throw std::runtime_error("Failed to create DiskANN native build input");
    }

    const uint32_t rows = static_cast<uint32_t>(size_);
    const uint32_t dim = params_.dim;
    out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char *>(&dim), sizeof(dim));

    std::vector<float> buffer(dim);
    for (size_t vid = 0; vid < size_; ++vid) {
      if (!reconstruct(vid, buffer.data())) {
        throw std::runtime_error(
            "Failed to reconstruct DiskANN point for native build");
      }
      out.write(reinterpret_cast<const char *>(buffer.data()),
                static_cast<std::streamsize>(dim * sizeof(float)));
    }
    if (!out) {
      throw std::runtime_error("Failed while writing DiskANN native build input");
    }
  }

  void load_native(const std::string &path) {
    // PQFlashIndex stores the reader as a reference to a shared_ptr, so it
    // must bind to the owning member rather than a local temporary.
    file_reader_ = make_aligned_reader();
    auto flash = std::make_unique<DiskannPQFlashT>(
        file_reader_, vec_to_diskann_metric(params_.metric_tag));
    const int rc = flash->load(params_.build_threads > 0
                                   ? static_cast<uint32_t>(params_.build_threads)
                                   : 1,
                               path.c_str());
    if (rc != 0) {
      throw std::runtime_error("DiskANN PQFlashIndex load failed");
    }

    const uint64_t cache_nodes =
        std::min<uint64_t>(flash->get_num_points(), kDefaultCacheNodes);
    if (cache_nodes > 0) {
      std::vector<uint32_t> node_list;
      flash->cache_bfs_levels(cache_nodes, node_list);
      if (!node_list.empty()) {
        flash->load_cache_list(node_list);
      }
    }

    memory_index_.reset();
    flash_index_ = std::move(flash);
    size_ = static_cast<size_t>(flash_index_->get_num_points());
    capacity_ = 0;
  }
};

}  // namespace

std::unique_ptr<IVectorIndex> vec_make_diskann_index(const vec_params_t &p) {
  if (p.type_tag != VEC_T_FLAT && p.type_tag != VEC_T_HNSW) {
    ib::warn() << "VECINDEX: DiskANN backend currently supports only FLAT/VAMANA"
               << " mutable/runtime modes.";
    return nullptr;
  }

  return std::make_unique<DiskannVectorIndex>(p);
}

bool vec_diskann_has_native_artifacts(const std::string &prefix) {
  if (prefix.empty()) {
    return false;
  }
  const std::string disk_index = prefix + "_disk.index";
  if (file_exists(disk_index)) {
    return true;
  }
  for (const char *suffix : kDiskannArtifactSuffixes) {
    if (file_exists(prefix + suffix)) {
      return true;
    }
  }
  return false;
}

void vec_diskann_collect_artifact_paths(const std::string &prefix,
                                        std::vector<std::string> *out) {
  if (out == nullptr) {
    return;
  }
  for (const auto &path : collect_diskann_artifacts(prefix)) {
    append_unique(out, path);
  }
}

void vec_diskann_remove_artifacts(const std::string &prefix) {
  remove_diskann_artifacts(prefix);
}

bool vec_diskann_build_from_fbin(const vec_params_t &params, size_t points_num,
                                 const std::string &data_path,
                                 const std::string &prefix,
                                 std::string *error_out) {
  if (error_out != nullptr) {
    error_out->clear();
  }
  if (data_path.empty() || prefix.empty() || points_num == 0) {
    if (error_out != nullptr) {
      *error_out = "empty DiskANN build input";
    }
    return false;
  }

  remove_diskann_artifacts(prefix);

  const std::string build_params =
      build_native_diskann_params(params, points_num);
  const int rc = diskann::build_disk_index<float>(
      data_path.c_str(), prefix.c_str(), build_params.c_str(),
      vec_to_diskann_metric(params.metric_tag), false);
  if (rc != 0) {
    remove_diskann_artifacts(prefix);
    if (error_out != nullptr) {
      *error_out = "DiskANN native build_disk_index failed";
    }
    return false;
  }

  return true;
}
