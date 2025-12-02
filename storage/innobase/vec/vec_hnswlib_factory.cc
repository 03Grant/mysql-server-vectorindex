#include "vec_hnswlib_factory.h"
#include "vec_index.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <omp.h>
#include <vector>

namespace {

std::unique_ptr<hnswlib::SpaceInterface<float>> make_space(const vec_params_t& p) {
  switch (p.metric_tag) {
    case VEC_M_IP:
    case VEC_M_COSINE:
      return std::make_unique<hnswlib::InnerProductSpace>(p.dim);
    case VEC_M_L2:
    default:
      return std::make_unique<hnswlib::L2Space>(p.dim);
  }
}

class HnswlibVectorIndex : public IVectorIndex {
 public:
  enum class IndexKind { Hnsw, Flat };

  HnswlibVectorIndex(vec_params_t params,
                     std::unique_ptr<hnswlib::SpaceInterface<float>> space,
                     std::unique_ptr<hnswlib::HierarchicalNSW<float>> index)
      : params_(params),
        kind_(IndexKind::Hnsw),
        space_(std::move(space)),
        hnsw_index_(std::move(index)) {}

  HnswlibVectorIndex(vec_params_t params,
                     std::unique_ptr<hnswlib::SpaceInterface<float>> space,
                     std::unique_ptr<hnswlib::BruteforceSearch<float>> index)
      : params_(params),
        kind_(IndexKind::Flat),
        space_(std::move(space)),
        bf_index_(std::move(index)) {}

  size_t dim() const override { return params_.dim; }

  void train(size_t, const float*) override {
    // HNSWLIB does not require explicit training.
  }

  void add(size_t n, const float* xb, const int64_t* ids) override {
    if (!index_ready()) return;
    ensure_capacity(n);

#ifdef _OPENMP
    const int threads = params_.build_threads > 0 ? params_.build_threads : 1;
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
      const size_t idx = static_cast<size_t>(i);
      const hnswlib::labeltype label =
          ids ? static_cast<hnswlib::labeltype>(ids[idx])
              : static_cast<hnswlib::labeltype>(current_count());
      active_index()->addPoint(xb + idx * params_.dim, label);
    }
  }

  void search(size_t nq, const float* xq, size_t k,
              int64_t* out_ids, float* out_distances) const override {
    if (!index_ready()) return;

    const bool prefer_small = !(params_.metric_tag == VEC_M_IP ||
                                params_.metric_tag == VEC_M_COSINE);

    for (size_t qi = 0; qi < nq; ++qi) {
      float* Dq = out_distances + qi * k;
      int64_t* Iq = out_ids + qi * k;

      // Fill defaults in case results < k.
      for (size_t t = 0; t < k; ++t) {
        Dq[t] = prefer_small ? std::numeric_limits<float>::infinity()
                             : -std::numeric_limits<float>::infinity();
        Iq[t] = -1;
      }

      auto result = active_index()->searchKnn(xq + qi * params_.dim, k);
      size_t pos = result.size();
      while (!result.empty() && pos > 0) {
        auto [dist, label] = result.top();
        result.pop();
        --pos;
        float score = dist;
        if (!prefer_small) {
          // InnerProduct/Cosine space returns distance; convert to similarity.
          score = 1.0f - dist;
        }
        Dq[pos] = score;
        Iq[pos] = static_cast<int64_t>(label);
      }
    }
  }

  void save(const std::string& path) const override {
    if (!index_ready()) return;
    // hnswlib::AlgorithmInterface lacks a const saveIndex; this is logically const.
    if (auto* idx = active_index_mutable()) {
      idx->saveIndex(path);
    }
  }

  void load(const std::string& path) override {
    // Recreate index over existing space.
    if (kind_ == IndexKind::Hnsw) {
      hnsw_index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(), path);
    } else {
      bf_index_ = std::make_unique<hnswlib::BruteforceSearch<float>>(space_.get(), path);
    }
  }

  size_t ntotal() const override {
    if (!index_ready()) return 0;
    return kind_ == IndexKind::Hnsw ? hnsw_index_->getCurrentElementCount()
                                    : bf_index_->cur_element_count;
  }

  bool reconstruct(size_t id, float* out) const override {
    if (!index_ready() || out == nullptr) return false;
    const hnswlib::labeltype label = static_cast<hnswlib::labeltype>(id);
    if (kind_ == IndexKind::Hnsw) {
      return false;  // HNSW reconstruct not implemented
    }

    auto *bf = bf_index_.get();
    if (bf == nullptr) {
      return false;
    }
    auto it = bf->dict_external_to_internal.find(label);
    if (it == bf->dict_external_to_internal.end()) {
      return false;
    }
    const size_t internal = it->second;
    const char *src = bf->data_ + bf->size_per_element_ * internal;
    std::memcpy(out, src, sizeof(float) * params_.dim);
    return true;
  }

  void set_search_params(const VecRuntimeSearchParams& params) override {
    if (!index_ready()) return;
    if (params.ef_search.has_value() && kind_ == IndexKind::Hnsw) {
      hnsw_index_->setEf(static_cast<size_t>(*params.ef_search));
    }
  }

 private:
  vec_params_t params_;
  IndexKind kind_;
  std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_index_;
  std::unique_ptr<hnswlib::BruteforceSearch<float>> bf_index_;

  void ensure_capacity(size_t incoming) {
    if (!index_ready()) return;
    if (kind_ == IndexKind::Hnsw) {
      ensure_capacity_hnsw(incoming);
    } else {
      ensure_capacity_flat(incoming);
    }
  }

  void ensure_capacity_hnsw(size_t incoming) {
    const size_t current = hnsw_index_->getCurrentElementCount();
    const size_t max_allowed = hnsw_index_->getMaxElements();
    if (current + incoming <= max_allowed) return;

    size_t new_cap = std::max(max_allowed * 2, current + incoming);
    // Avoid zero cap if misconfigured size=0.
    if (new_cap == 0) new_cap = std::max<size_t>(incoming, 1024);
    hnsw_index_->resizeIndex(new_cap);
  }

  void ensure_capacity_flat(size_t incoming) {
    // BruteforceSearch has a fixed allocation; rebuild with larger capacity when needed.
    const size_t current = bf_index_->cur_element_count;
    const size_t max_allowed = bf_index_->maxelements_;
    if (current + incoming <= max_allowed) return;

    size_t new_cap = std::max(max_allowed * 2, current + incoming);
    if (new_cap == 0) new_cap = std::max<size_t>(incoming, 1024);

    auto new_index = std::make_unique<hnswlib::BruteforceSearch<float>>(space_.get(), new_cap);
    new_index->cur_element_count = current;
    new_index->dict_external_to_internal = bf_index_->dict_external_to_internal;
    const size_t copy_bytes = current * bf_index_->size_per_element_;
    if (copy_bytes > 0) {
      std::memcpy(new_index->data_, bf_index_->data_, copy_bytes);
    }
    bf_index_ = std::move(new_index);
  }

  bool index_ready() const {
    return (kind_ == IndexKind::Hnsw && hnsw_index_ != nullptr) ||
           (kind_ == IndexKind::Flat && bf_index_ != nullptr);
  }

  hnswlib::AlgorithmInterface<float>* active_index() {
    return kind_ == IndexKind::Hnsw
               ? static_cast<hnswlib::AlgorithmInterface<float>*>(hnsw_index_.get())
               : static_cast<hnswlib::AlgorithmInterface<float>*>(bf_index_.get());
  }

  const hnswlib::AlgorithmInterface<float>* active_index() const {
    return kind_ == IndexKind::Hnsw
               ? static_cast<const hnswlib::AlgorithmInterface<float>*>(hnsw_index_.get())
               : static_cast<const hnswlib::AlgorithmInterface<float>*>(bf_index_.get());
  }

  hnswlib::AlgorithmInterface<float>* active_index_mutable() const {
    return kind_ == IndexKind::Hnsw
               ? const_cast<hnswlib::AlgorithmInterface<float>*>(
                     static_cast<const hnswlib::AlgorithmInterface<float>*>(hnsw_index_.get()))
               : const_cast<hnswlib::AlgorithmInterface<float>*>(
                     static_cast<const hnswlib::AlgorithmInterface<float>*>(bf_index_.get()));
  }

  size_t current_count() const {
    return kind_ == IndexKind::Hnsw ? hnsw_index_->getCurrentElementCount()
                                    : bf_index_->cur_element_count;
  }
};

}  // namespace

std::unique_ptr<IVectorIndex> vec_make_hnswlib_index(const vec_params_t& p) {
  if (p.type_tag != VEC_T_HNSW && p.type_tag != VEC_T_FLAT) {
    return nullptr; // hnswlib 只支持 HNSW 或 Flat (bruteforce)
  }
  auto space = make_space(p);
  if (p.type_tag == VEC_T_FLAT) {
    auto index = std::make_unique<hnswlib::BruteforceSearch<float>>(space.get(), p.size);
    return std::make_unique<HnswlibVectorIndex>(p, std::move(space), std::move(index));
  }

  auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
      space.get(), p.size, p.hnsw_m, p.efConstruction);
  index->setEf(std::max(static_cast<size_t>(p.efConstruction), static_cast<size_t>(50)));
  return std::make_unique<HnswlibVectorIndex>(p, std::move(space), std::move(index));
}
