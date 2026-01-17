#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

struct VecRuntimeSearchParams {
  std::optional<size_t> nprobe;    // IVF
  std::optional<int>    ef_search; // HNSW
};

// Backend-agnostic vector index interface so different libraries (Faiss, hnswlib, ...)
// can be plugged in behind a single API.
class IVectorIndex {
public:
  virtual ~IVectorIndex() = default;

  virtual size_t dim() const = 0;

  // For implementations without training, leave empty.
  virtual void train(size_t n, const float* xb) = 0;
  virtual void add(size_t n, const float* xb, const int64_t* ids) = 0;

  virtual void search(size_t nq, const float* xq, size_t k,
                      int64_t* out_ids, float* out_distances,
                      const VecRuntimeSearchParams* params) const = 0;

  virtual void save(const std::string& path) const = 0;
  virtual void load(const std::string& path) = 0;

  virtual size_t ntotal() const = 0;
  virtual bool reconstruct(size_t id, float* out) const = 0;

  // Optional runtime tuning knobs (ignored if backend/type does not support).
  virtual void set_search_params(const VecRuntimeSearchParams& params) = 0;
};
