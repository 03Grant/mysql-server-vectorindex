// vec_faiss_factory.cc
#include "vec_faiss_includes.h"
#include "vec_faiss_factory.h"

static inline faiss::MetricType vec_to_faiss_metric(uint8_t metric_tag) {
  switch (metric_tag) {
    case VEC_M_IP:     return faiss::METRIC_INNER_PRODUCT;
    case VEC_M_COSINE: return faiss::METRIC_INNER_PRODUCT; // 训练/查询时自行单位化；构造仍用 IP
    case VEC_M_L2:
    default:           return faiss::METRIC_L2;
  }
}

std::unique_ptr<faiss::Index> vec_make_faiss_index(const vec_params_t& p) {
  const faiss::MetricType metric = vec_to_faiss_metric(p.metric_tag);

  switch (p.type_tag) {
    case VEC_T_FLAT: {
      return std::make_unique<faiss::IndexFlat>(p.dim, metric);
    }

    case VEC_T_IVFFLAT: {
      // 注意：IVF 构造函数不会接管 unique_ptr 所有权；
      // 若让 IVF 负责销毁量化器，需传 raw* 并设置 own_fields=true。
      auto* quant = new faiss::IndexFlat(p.dim, metric);
      auto ivf = std::make_unique<faiss::IndexIVFFlat>(quant, p.dim, p.nlist, metric);
      ivf->own_fields = true; // IVF 负责 delete quant
      return ivf;
    }

    case VEC_T_IVFPQ: {
      auto* quant = new faiss::IndexFlat(p.dim, metric);
      // 你在解析里已保证 nbits*m | dim（或至少合法性）
      auto ivfpq = std::make_unique<faiss::IndexIVFPQ>(
          quant, p.dim, p.nlist, p.m, p.nbits, metric);
      ivfpq->own_fields = true;
      return ivfpq;
    }

    case VEC_T_HNSW: {
      auto h = std::make_unique<faiss::IndexHNSWFlat>(p.dim, p.hnsw_m, metric);
      h->hnsw.efConstruction = p.efConstruction;
      return h;
    }

    default:
      // fallback：给个 FLAT，保证不崩
      return std::make_unique<faiss::IndexFlat>(p.dim, metric);
  }
}