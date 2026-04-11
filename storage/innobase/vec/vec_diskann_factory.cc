#include "vec_diskann_factory.h"

#include <immintrin.h>
#include "index.h"
#include "parameters.h"
#include "distance.h"
#include "ut0ut.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using DiskannIndexT = diskann::Index<float, uint32_t, uint32_t>;

constexpr uint32_t kDiskannTagOffset = 1;

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
  if (id < 0 || id >= static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
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

class DiskannVectorIndex : public IVectorIndex {
 public:
  explicit DiskannVectorIndex(const vec_params_t &params)
      : params_(params),
        write_params_(diskann::IndexWriteParametersBuilder(
                          choose_diskann_search_l(params),
                          choose_diskann_degree(params))
                          .with_num_threads(params.build_threads > 0
                                                ? static_cast<uint32_t>(params.build_threads)
                                                : 1)
                          .build()),
        search_params_(choose_diskann_search_l(params),
                       params.build_threads > 0
                           ? static_cast<uint32_t>(params.build_threads)
                           : 1),
        default_search_l_(search_params_.initial_search_list_size),
        capacity_(choose_diskann_capacity(params_)) {
    index_ = make_index(capacity_);
  }

  size_t dim() const override { return params_.dim; }

  void train(size_t, const float *) override {
    // Dynamic memory DiskANN does not require an explicit train step.
  }

  void add(size_t n, const float *xb, const int64_t *ids) override {
    if (index_ == nullptr || xb == nullptr || n == 0) {
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
      const int rc = index_->insert_point(xb + i * params_.dim, tag);
      if (rc != 0) {
        throw std::runtime_error("DiskANN insert_point failed");
      }
      size_ = std::max<size_t>(size_, static_cast<size_t>(vid + 1));
    }
  }

  void search(size_t nq, const float *xq, size_t k, int64_t *out_ids,
              float *out_distances,
              const VecRuntimeSearchParams *params) const override {
    if (index_ == nullptr || xq == nullptr || out_ids == nullptr ||
        out_distances == nullptr) {
      return;
    }

    const bool prefer_small = diskann_metric_prefers_small(params_.metric_tag);
    const uint32_t search_l = choose_runtime_search_l(params, k);

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
      const size_t found = index_->search_with_tags(
          xq + qi * params_.dim, k_use, search_l, tags.data(),
          distances.data(), res_vectors);

      for (size_t j = 0; j < found; ++j) {
        i_row[j] = tag_to_vid(tags[j]);
        d_row[j] = prefer_small ? distances[j] : (1.0f - distances[j]);
      }
    }
  }

  void save(const std::string &path) const override {
    if (index_ != nullptr) {
      index_->save(path.c_str());
    }
  }

  void load(const std::string &path) override {
    const size_t frozen_pts = DiskannIndexT::get_graph_num_frozen_points(path);
    index_ = make_index(0, frozen_pts);
    index_->load(path.c_str(),
                 params_.build_threads > 0
                     ? static_cast<uint32_t>(params_.build_threads)
                     : 1,
                 default_search_l_);
    size_ = index_->get_num_points();
  }

  size_t ntotal() const override { return size_; }

  bool reconstruct(size_t id, float *out) const override {
    if (index_ == nullptr || out == nullptr) {
      return false;
    }
    uint32_t tag;
    try {
      tag = vid_to_tag(static_cast<int64_t>(id));
    } catch (...) {
      return false;
    }
    return index_->get_vector_by_tag(tag, out) == 0;
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
  std::unique_ptr<DiskannIndexT> index_;
  size_t size_{0};
  uint32_t default_search_l_{64};
  size_t capacity_{0};

  std::unique_ptr<DiskannIndexT> make_index(size_t capacity,
                                            size_t num_frozen_pts = 1) const {
    const size_t effective_capacity = capacity > 0 ? capacity : 0;
    return std::make_unique<DiskannIndexT>(
        vec_to_diskann_metric(params_.metric_tag), params_.dim, effective_capacity,
        std::make_shared<diskann::IndexWriteParameters>(write_params_),
        std::make_shared<diskann::IndexSearchParams>(search_params_),
        num_frozen_pts, true, true, false, false, 0, false, false);
  }

  void bulk_build(size_t n, const float *xb, const int64_t *ids) {
    std::vector<uint32_t> tags;
    tags.reserve(n);
    size_t max_vid = 0;

    for (size_t i = 0; i < n; ++i) {
      const int64_t vid =
          ids != nullptr ? ids[i] : static_cast<int64_t>(i);
      tags.push_back(vid_to_tag(vid));
      max_vid = std::max(max_vid, static_cast<size_t>(vid + 1));
    }

    index_->build(xb, n, tags);
    size_ = max_vid;
  }

  void ensure_capacity(size_t incoming) {
    if (index_ == nullptr) {
      return;
    }
    const size_t needed = size_ + incoming;
    if (needed <= capacity_) {
      return;
    }

    size_t new_capacity = capacity_ > 0 ? capacity_ : choose_diskann_capacity(params_);
    while (new_capacity < needed) {
      new_capacity = std::max(new_capacity * 2, needed);
    }

    auto rebuilt = make_index(new_capacity);
    std::vector<float> buffer(params_.dim);
    for (size_t vid = 0; vid < size_; ++vid) {
      if (!reconstruct(vid, buffer.data())) {
        throw std::runtime_error("DiskANN rebuild failed during capacity growth");
      }
      const uint32_t tag = vid_to_tag(static_cast<int64_t>(vid));
      if (rebuilt->insert_point(buffer.data(), tag) != 0) {
        throw std::runtime_error("DiskANN rebuild insert failed");
      }
    }

    index_ = std::move(rebuilt);
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
