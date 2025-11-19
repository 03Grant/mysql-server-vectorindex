#include "vec_hnswlib_factory.h"
#include "vec_index.h"
#include <algorithm>
#include <limits>
#include <memory>
#include <omp.h>

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
  explicit HnswlibVectorIndex(vec_params_t params,
                              std::unique_ptr<hnswlib::SpaceInterface<float>> space,
                              std::unique_ptr<hnswlib::HierarchicalNSW<float>> index)
      : params_(params),
        space_(std::move(space)),
        index_(std::move(index)) {}

  size_t dim() const override { return params_.dim; }

  void train(size_t, const float*) override {
    // HNSWLIB does not require explicit training.
  }

  void add(size_t n, const float* xb, const int64_t* ids) override {
    if (!index_) return;
    ensure_capacity(n);

#ifdef _OPENMP
    const int threads = params_.build_threads > 0 ? params_.build_threads : 1;
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
      const hnswlib::labeltype label =
          ids ? static_cast<hnswlib::labeltype>(ids[i])
              : static_cast<hnswlib::labeltype>(index_->getCurrentElementCount());
      index_->addPoint(xb + static_cast<size_t>(i) * params_.dim, label);
    }
  }

  void search(size_t nq, const float* xq, size_t k,
              int64_t* out_ids, float* out_distances) const override {
    if (!index_) return;

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

      auto result = index_->searchKnn(xq + qi * params_.dim, k);
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
    if (index_) index_->saveIndex(path);
  }

  void load(const std::string& path) override {
    // Recreate index over existing space.
    index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(), path);
  }

  size_t ntotal() const override {
    return index_ ? index_->getCurrentElementCount() : 0;
  }

  void set_search_params(const VecRuntimeSearchParams& params) override {
    if (!index_) return;
    if (params.ef_search.has_value()) {
      index_->setEf(static_cast<size_t>(*params.ef_search));
    }
  }

 private:
  vec_params_t params_;
  std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;

  void ensure_capacity(size_t incoming) {
    if (!index_) return;
    const size_t current = index_->getCurrentElementCount();
    const size_t max_allowed = index_->getMaxElements();
    if (current + incoming <= max_allowed) return;

    size_t new_cap = std::max(max_allowed * 2, current + incoming);
    // Avoid zero cap if misconfigured size=0.
    if (new_cap == 0) new_cap = std::max<size_t>(incoming, 1024);
    index_->resizeIndex(new_cap);
  }
};

}  // namespace

std::unique_ptr<IVectorIndex> vec_make_hnswlib_index(const vec_params_t& p) {
  if (p.type_tag != VEC_T_HNSW) {
    return nullptr; // hnswlib 暂时只支持 HNSW
  }
  auto space = make_space(p);
  auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
      space.get(), p.size, p.hnsw_m, p.efConstruction);
  index->setEf(std::max(static_cast<size_t>(p.efConstruction), static_cast<size_t>(50)));
  return std::make_unique<HnswlibVectorIndex>(p, std::move(space), std::move(index));
}
