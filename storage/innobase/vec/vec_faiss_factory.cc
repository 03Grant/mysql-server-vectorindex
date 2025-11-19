// vec_faiss_factory.cc
#include "vec_faiss_includes.h"
#include "vec_faiss_factory.h"
#include "vec_index.h"
#include <faiss/index_io.h>
#include <vector>
#include <type_traits>

static inline faiss::MetricType vec_to_faiss_metric(uint8_t metric_tag) {
  switch (metric_tag) {
    case VEC_M_IP:     return faiss::METRIC_INNER_PRODUCT;
    case VEC_M_COSINE: return faiss::METRIC_INNER_PRODUCT; // 训练/查询时自行单位化；构造仍用 IP
    case VEC_M_L2:
    default:           return faiss::METRIC_L2;
  }
}

namespace {
class FaissVectorIndex : public IVectorIndex {
 public:
  explicit FaissVectorIndex(std::unique_ptr<faiss::Index> index)
      : index_(std::move(index)) {}

  size_t dim() const override { return index_ ? index_->d : 0; }

  void train(size_t n, const float* xb) override {
    if (index_ && !index_->is_trained) index_->train(n, xb);
  }

  void add(size_t n, const float* xb, const int64_t* ids) override {
    if (!index_) return;
    if constexpr (std::is_same_v<faiss::idx_t, int64_t>) {
      if (ids != nullptr) {
        index_->add(n, xb);;
      } else {
        index_->add(n, xb);
      }
    } else {
      if (ids != nullptr) {
        std::vector<faiss::idx_t> faiss_ids(n);
        for (size_t i = 0; i < n; ++i) faiss_ids[i] = static_cast<faiss::idx_t>(ids[i]);
        index_->add(n, xb);
      } else {
        index_->add(n, xb);
      }
    }
  }

  void search(size_t nq, const float* xq, size_t k,
              int64_t* out_ids, float* out_distances) const override {
    if (!index_) return;
    if constexpr (std::is_same_v<faiss::idx_t, int64_t>) {
      index_->search(nq, xq, k, out_distances,
                     reinterpret_cast<faiss::idx_t*>(out_ids));
    } else {
      std::vector<faiss::idx_t> tmp_ids(nq * k);
      index_->search(nq, xq, k, out_distances, tmp_ids.data());
      for (size_t i = 0; i < nq * k; ++i) {
        out_ids[i] = static_cast<int64_t>(tmp_ids[i]);
      }
    }
  }

  void save(const std::string& path) const override {
    if (index_) faiss::write_index(index_.get(), path.c_str());
  }

  void load(const std::string& path) override {
    index_.reset(faiss::read_index(path.c_str()));
  }

  size_t ntotal() const override {
    return index_ ? static_cast<size_t>(index_->ntotal) : 0;
  }

  void set_search_params(const VecRuntimeSearchParams& params) override {
    if (!index_) return;

    if (params.nprobe.has_value()) {
      if (auto* ivf = dynamic_cast<faiss::IndexIVF*>(index_.get())) {
        ivf->nprobe = static_cast<faiss::idx_t>(*params.nprobe);
      }
    }
    if (params.ef_search.has_value()) {
      if (auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(index_.get())) {
        hnsw->hnsw.efSearch = *params.ef_search;
      }
    }
  }

 private:
  std::unique_ptr<faiss::Index> index_;
};
} // namespace

std::unique_ptr<IVectorIndex> vec_make_faiss_index(const vec_params_t& p) {
  const faiss::MetricType metric = vec_to_faiss_metric(p.metric_tag);
  std::unique_ptr<faiss::Index> index;

  switch (p.type_tag) {
    case VEC_T_FLAT: {
      index = std::make_unique<faiss::IndexFlat>(p.dim, metric);
      break;
    }

    case VEC_T_IVFFLAT: {
      // 注意：IVF 构造函数不会接管 unique_ptr 所有权；
      // 若让 IVF 负责销毁量化器，需传 raw* 并设置 own_fields=true。
      auto* quant = new faiss::IndexFlat(p.dim, metric);
      auto ivf = std::make_unique<faiss::IndexIVFFlat>(quant, p.dim, p.nlist, metric);
      ivf->own_fields = true; // IVF 负责 delete quant
      index = std::move(ivf);
      break;
    }

    case VEC_T_IVFPQ: {
      auto* quant = new faiss::IndexFlat(p.dim, metric);
      auto ivfpq = std::make_unique<faiss::IndexIVFPQ>(
          quant, p.dim, p.nlist, p.m, p.nbits, metric);
      ivfpq->own_fields = true;
      index = std::move(ivfpq);
      break;
    }

    case VEC_T_HNSW: {
      auto h = std::make_unique<faiss::IndexHNSWFlat>(p.dim, p.hnsw_m, metric);
      h->hnsw.efConstruction = p.efConstruction;
      index = std::move(h);
      break;
    }

    default:
      // fallback：给个 FLAT，保证不崩
      index = std::make_unique<faiss::IndexFlat>(p.dim, metric);
      break;
  }

  return std::make_unique<FaissVectorIndex>(std::move(index));
}
