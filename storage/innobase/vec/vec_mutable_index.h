#pragma once
// vec_mutable_index.h
//
// MutableFlatIndex: an append-only, brute-force flat index used for the LSM
// mutable segment. It is designed for lock-free reads concurrent with appends:
//
//   * Vectors live in fixed-size chunks held in a fixed-size pointer array, so
//     an existing entry is never relocated by a later append.
//   * An atomic watermark (committed_) publishes only fully-written entries.
//     A reader loads it with acquire and scans [0, n); a writer fills slot n
//     then stores n+1 with release, so the reader never observes a half-written
//     entry and never reads a chunk pointer before it is set.
//
// Concurrency contract:
//   * add()  -- writers must be serialized by the caller (the per-segment
//               append latch). add() must be the LAST step of an insert so the
//               watermark publishes only after the row's TID/PK are written.
//   * search()/reconstruct()/ntotal() -- lock-free; safe concurrently with
//     add() and with each other.
//
// Distances match Faiss IndexFlat semantics so results merge correctly with the
// immutable segments: METRIC_L2 returns squared L2 (smaller is better),
// METRIC_INNER_PRODUCT returns the inner product (larger is better). Cosine is
// inner product on externally-normalized vectors, same as the Faiss path.

#include "vec_index.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class MutableFlatIndex : public IVectorIndex {
 public:
  // metric_tag values match vec_params.h VEC_M_*: 0=L2, 1=IP, 2=COSINE.
  static constexpr uint8_t kMetricL2 = 0;
  static constexpr uint8_t kMetricIP = 1;
  static constexpr uint8_t kMetricCosine = 2;

  // Capacity = kChunkEntries * kMaxChunks entries. The mutable segment is
  // sealed (rotated) at params.size, so this only has to cover size plus the
  // inserts that arrive before rotation completes. The pointer array costs
  // kMaxChunks * sizeof(ptr); chunks themselves are allocated lazily.
  static constexpr size_t kChunkEntries = 1024;
  static constexpr size_t kMaxChunks = 8192;  // 8.39M entries cap

  // Transaction id stored inline per entry. Equals InnoDB trx_id_t (uint64_t);
  // kept as uint64_t here so the header has no InnoDB dependency.
  using tid_t = uint64_t;

  MutableFlatIndex(size_t dim, uint8_t metric_tag)
      : dim_(dim), metric_tag_(metric_tag) {}

  size_t dim() const override { return dim_; }

  void train(size_t, const float*) override {}  // flat: no training

  bool holds_inline_meta() const override { return true; }

  // Append n vectors WITHOUT inline metadata. Caller must hold the per-segment
  // append latch. `ids` is ignored: the assigned id is the sequential slot,
  // matching Faiss IndexFlat. Used by paths that don't carry TID/PK (e.g. tests
  // and the bulk no-aux recovery path); get_trx_id/get_pk then return empty.
  void add(size_t n, const float* xb, const int64_t* /*ids*/) override {
    if (xb == nullptr) return;
    for (size_t i = 0; i < n; ++i) {
      if (!write_slot(committed_.load(std::memory_order_relaxed),
                      xb + i * dim_, 0, nullptr, 0)) {
        return;  // overflow
      }
    }
  }

  // Append one entry with its creator transaction id and primary-key bytes,
  // publishing all three together. Caller must hold the per-segment append
  // latch and must call this as the LAST step of an insert so a reader that
  // observes the published watermark also observes the vector, TID and PK.
  // Returns the assigned slot id, or SIZE_MAX on capacity overflow.
  size_t append(const float* vec, tid_t tid, const unsigned char* pk,
                size_t pk_len) {
    const size_t slot = committed_.load(std::memory_order_relaxed);
    if (!write_slot(slot, vec, tid, pk, pk_len)) {
      return ~size_t{0};
    }
    return slot;
  }

  // Lock-free reads of the inline metadata. Valid for slot < ntotal().
  tid_t get_trx_id(size_t slot) const {
    if (slot >= committed_.load(std::memory_order_acquire)) return 0;
    const Meta* m = meta_ptr(slot, /*allocate=*/false);
    return m ? m->tid : 0;
  }

  // Returns a pointer to the primary-key bytes for `slot` and sets *len. The
  // pointer is valid for the lifetime of this index (the entry is never
  // rewritten). Returns nullptr if slot is unpublished or has no PK.
  const unsigned char* get_pk(size_t slot, size_t* len) const {
    if (len) *len = 0;
    if (slot >= committed_.load(std::memory_order_acquire)) return nullptr;
    const Meta* m = meta_ptr(slot, /*allocate=*/false);
    if (m == nullptr || m->pk.empty()) return nullptr;
    if (len) *len = m->pk.size();
    return m->pk.data();
  }

  size_t ntotal() const override {
    return committed_.load(std::memory_order_acquire);
  }

  bool reconstruct(size_t id, float* out) const override {
    if (out == nullptr) return false;
    const size_t n = committed_.load(std::memory_order_acquire);
    if (id >= n) return false;
    const float* src = slot_ptr(id, /*allocate=*/false);
    if (src == nullptr) return false;
    std::memcpy(out, src, dim_ * sizeof(float));
    return true;
  }

  void search(size_t nq, const float* xq, size_t k, int64_t* out_ids,
              float* out_distances,
              const VecRuntimeSearchParams* /*params*/) const override {
    const bool prefer_small = !(metric_tag_ == kMetricIP ||
                                metric_tag_ == kMetricCosine);
    const size_t n = committed_.load(std::memory_order_acquire);

    for (size_t q = 0; q < nq; ++q) {
      const float* query = xq + q * dim_;
      int64_t* ids = out_ids + q * k;
      float* dists = out_distances + q * k;

      std::vector<std::pair<float, int64_t>> cand;
      cand.reserve(n);
      for (size_t i = 0; i < n; ++i) {
        const float* v = slot_ptr(i, /*allocate=*/false);
        if (v == nullptr) continue;
        const float d = prefer_small ? l2sq(query, v) : ip(query, v);
        cand.emplace_back(d, static_cast<int64_t>(i));
      }

      const size_t m = std::min(k, cand.size());
      if (prefer_small) {
        std::partial_sort(
            cand.begin(), cand.begin() + m, cand.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
      } else {
        std::partial_sort(
            cand.begin(), cand.begin() + m, cand.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
      }

      for (size_t j = 0; j < k; ++j) {
        if (j < m) {
          dists[j] = cand[j].first;
          ids[j] = cand[j].second;
        } else {
          dists[j] = prefer_small ? std::numeric_limits<float>::infinity()
                                  : -std::numeric_limits<float>::infinity();
          ids[j] = -1;
        }
      }
    }
  }

  // The mutable segment is never persisted directly; it is rebuilt into an
  // immutable index on flush (via ntotal()+reconstruct()) and recovered from
  // the auxiliary table on restart. save/load are therefore no-ops.
  void save(const std::string&) const override {}
  void load(const std::string&) override {}
  void set_search_params(const VecRuntimeSearchParams&) override {}

  bool overflowed() const { return overflow_; }

 private:
  struct Meta {
    tid_t tid{0};
    std::vector<unsigned char> pk;
  };

  // Write vector (+ optional inline meta) to `slot` and publish the watermark.
  // Returns false on capacity overflow.
  bool write_slot(size_t slot, const float* vec, tid_t tid,
                  const unsigned char* pk, size_t pk_len) {
    float* dst = slot_ptr(slot, /*allocate=*/true);
    if (dst == nullptr || vec == nullptr) {
      overflow_ = (dst == nullptr);
      return false;
    }
    Meta* m = meta_ptr(slot, /*allocate=*/true);
    if (m == nullptr) {
      overflow_ = true;
      return false;
    }
    std::memcpy(dst, vec, dim_ * sizeof(float));
    m->tid = tid;
    if (pk != nullptr && pk_len > 0) {
      m->pk.assign(pk, pk + pk_len);
    } else {
      m->pk.clear();
    }
    // Publish: every write above happens-before this release store, so a reader
    // that loads the watermark with acquire sees the vector, TID and PK.
    committed_.store(slot + 1, std::memory_order_release);
    return true;
  }

  float* slot_ptr(size_t slot, bool allocate) const {
    const size_t c = slot / kChunkEntries;
    if (c >= kMaxChunks) return nullptr;
    if (!chunks_[c]) {
      if (!allocate) return nullptr;
      chunks_[c] = std::make_unique<float[]>(kChunkEntries * dim_);
    }
    return chunks_[c].get() + (slot % kChunkEntries) * dim_;
  }

  Meta* meta_ptr(size_t slot, bool allocate) const {
    const size_t c = slot / kChunkEntries;
    if (c >= kMaxChunks) return nullptr;
    if (!meta_chunks_[c]) {
      if (!allocate) return nullptr;
      meta_chunks_[c] = std::make_unique<Meta[]>(kChunkEntries);
    }
    return &meta_chunks_[c][slot % kChunkEntries];
  }

  float l2sq(const float* a, const float* b) const {
    float s = 0.0f;
    for (size_t d = 0; d < dim_; ++d) {
      const float x = a[d] - b[d];
      s += x * x;
    }
    return s;
  }

  float ip(const float* a, const float* b) const {
    float s = 0.0f;
    for (size_t d = 0; d < dim_; ++d) s += a[d] * b[d];
    return s;
  }

  const size_t dim_;
  const uint8_t metric_tag_;
  // Fixed-size pointer arrays => never relocate. Chunks are filled lazily by the
  // single appending writer; readers only dereference chunks for published
  // entries, synchronized via the committed_ release/acquire. chunks_ holds the
  // vectors; meta_chunks_ holds the per-entry inline TID/PK.
  mutable std::unique_ptr<float[]> chunks_[kMaxChunks];
  mutable std::unique_ptr<Meta[]> meta_chunks_[kMaxChunks];
  std::atomic<size_t> committed_{0};
  bool overflow_{false};
};
