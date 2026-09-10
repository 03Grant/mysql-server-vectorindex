#include "vec_hnswlib_factory.h"
#include "vec_index.h"
// univ.i must precede ut0ut.h: ut0ut.h defines the ut_is_2pow macro only after
// its own include chain pulls in ut0byte.ic (which uses it). Establishing the
// InnoDB header chain via univ.i first avoids that ordering error. (Pre-existing
// issue, unrelated to the versioned-snapshot change.)
#include "univ.i"
#include "ut0ut.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <omp.h>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

std::atomic<bool> g_hnsw_add_omp_logged{false};

#ifdef _OPENMP
void log_hnsw_add_omp_once(const vec_params_t& params, size_t n) {
  bool expected = false;
  if (!g_hnsw_add_omp_logged.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    return;
  }

  const int requested_threads = params.build_threads > 0 ? params.build_threads : 1;
  ib::warn() << "VECINDEX: HNSWlib add OpenMP requested_threads="
             << requested_threads
             << " actual_threads=" << omp_get_num_threads()
             << " max_threads=" << omp_get_max_threads()
             << " num_procs=" << omp_get_num_procs()
             << " dynamic=" << omp_get_dynamic()
             << " n=" << n
             << " dim=" << params.dim;
}
#endif

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

/* Fast deserializer for HierarchicalNSW, field-for-field equivalent to the
stock loadIndex() (same file format, produced by the stock saveIndex(); the
stock loader still reads these files). The stock implementation makes two
extra passes over the link-list tail with per-element stream reads and seeks
(each seek invalidating the ifstream buffer) and fills label_lookup_ without
reserving it; at 10M+ elements that overhead dominates recovery time. Here
the level0 block and the tail are read with one bulk read each and the tail
is parsed in memory; the stock integrity pre-scan is subsumed by exact
bounds checking during the parse. All HierarchicalNSW members are public, so
this lives entirely in the integration layer: the vendored library is
unmodified. Expects a default-constructed index (HierarchicalNSW(s)). */
void vec_fast_hnsw_load(hnswlib::HierarchicalNSW<float>* idx,
                        hnswlib::SpaceInterface<float>* s,
                        const std::string& location) {
  std::ifstream input(location, std::ios::binary);
  if (!input.is_open())
    throw std::runtime_error("Cannot open file");

  idx->clear();
  input.seekg(0, input.end);
  const std::streampos total_filesize = input.tellg();
  input.seekg(0, input.beg);

  hnswlib::readBinaryPOD(input, idx->offsetLevel0_);
  hnswlib::readBinaryPOD(input, idx->max_elements_);
  hnswlib::readBinaryPOD(input, idx->cur_element_count);

  const size_t cur_element_count = idx->cur_element_count;
  size_t max_elements = 0;  // no caller-side cap, exactly as our old call site
  if (max_elements < cur_element_count)
    max_elements = idx->max_elements_;
  idx->max_elements_ = max_elements;
  hnswlib::readBinaryPOD(input, idx->size_data_per_element_);
  hnswlib::readBinaryPOD(input, idx->label_offset_);
  hnswlib::readBinaryPOD(input, idx->offsetData_);
  hnswlib::readBinaryPOD(input, idx->maxlevel_);
  hnswlib::readBinaryPOD(input, idx->enterpoint_node_);

  hnswlib::readBinaryPOD(input, idx->maxM_);
  hnswlib::readBinaryPOD(input, idx->maxM0_);
  hnswlib::readBinaryPOD(input, idx->M_);
  hnswlib::readBinaryPOD(input, idx->mult_);
  hnswlib::readBinaryPOD(input, idx->ef_construction_);

  idx->data_size_ = s->get_data_size();
  idx->fstdistfunc_ = s->get_dist_func();
  idx->dist_func_param_ = s->get_dist_func_param();

  const auto pos = input.tellg();
  const size_t level0_bytes =
      cur_element_count * idx->size_data_per_element_;
  if (static_cast<size_t>(total_filesize) <
      static_cast<size_t>(pos) + level0_bytes)
    throw std::runtime_error("Index seems to be corrupted or unsupported");
  const size_t tail_bytes = static_cast<size_t>(total_filesize) -
                            static_cast<size_t>(pos) - level0_bytes;
  input.close();

  idx->data_level0_memory_ =
      (char*)malloc(max_elements * idx->size_data_per_element_);
  if (idx->data_level0_memory_ == nullptr)
    throw std::runtime_error(
        "Not enough memory: loadIndex failed to allocate level0");

  idx->size_links_per_element_ =
      idx->maxM_ * sizeof(hnswlib::tableint) + sizeof(hnswlib::linklistsizeint);
  idx->size_links_level0_ =
      idx->maxM0_ * sizeof(hnswlib::tableint) + sizeof(hnswlib::linklistsizeint);
  std::vector<std::mutex>(max_elements).swap(idx->link_list_locks_);
  std::vector<std::mutex>(
      hnswlib::HierarchicalNSW<float>::MAX_LABEL_OPERATION_LOCKS)
      .swap(idx->label_op_locks_);

  idx->visited_list_pool_.reset(new hnswlib::VisitedListPool(1, max_elements));

  idx->linkLists_ = (char**)malloc(sizeof(void*) * max_elements);
  if (idx->linkLists_ == nullptr)
    throw std::runtime_error(
        "Not enough memory: loadIndex failed to allocate linklists");
  idx->element_levels_ = std::vector<int>(max_elements);
  idx->revSize_ = 1.0 / idx->mult_;
  idx->ef_ = 10;
  idx->label_lookup_.reserve(cur_element_count);

  /* The level0 block is the bulk of the file (data + level0 links + labels;
  ~780 B per element). A single sequential read tops out well below the page
  cache's aggregate bandwidth, so it is read as fixed chunks by a small
  reader pool via pread. label_lookup_ construction only needs the labels of
  chunks already in memory, so a builder thread consumes finished chunks
  concurrently instead of making a separate pass afterwards. */
  std::unique_ptr<char[]> tail(new char[tail_bytes]);
  const int fd = ::open(location.c_str(), O_RDONLY);
  if (fd < 0)
    throw std::runtime_error("Cannot open file");

  {
    const size_t n_chunks = 32;
    const size_t chunk_bytes =
        ((level0_bytes + n_chunks - 1) / n_chunks + idx->size_data_per_element_ -
         1) /
        idx->size_data_per_element_ * idx->size_data_per_element_;
    const unsigned n_readers =
        std::min<unsigned>(8, std::max<unsigned>(
                                  1, std::thread::hardware_concurrency() / 2));

    std::mutex mu;
    std::condition_variable cv;
    std::vector<bool> chunk_done(n_chunks, false);
    std::atomic<size_t> next_chunk{0};
    std::atomic<bool> read_failed{false};

    auto reader_fn = [&]() {
      for (;;) {
        const size_t c = next_chunk.fetch_add(1);
        if (c >= n_chunks || read_failed.load(std::memory_order_relaxed))
          return;
        const size_t begin = c * chunk_bytes;
        if (begin >= level0_bytes) {
          std::lock_guard<std::mutex> lk(mu);
          chunk_done[c] = true;
          cv.notify_all();
          continue;
        }
        const size_t want = std::min(chunk_bytes, level0_bytes - begin);
        size_t got = 0;
        while (got < want) {
          const ssize_t n = ::pread(
              fd, idx->data_level0_memory_ + begin + got, want - got,
              static_cast<off_t>(pos) + static_cast<off_t>(begin + got));
          if (n <= 0) {
            read_failed.store(true, std::memory_order_relaxed);
            break;
          }
          got += static_cast<size_t>(n);
        }
        std::lock_guard<std::mutex> lk(mu);
        chunk_done[c] = true;
        cv.notify_all();
      }
    };

    /* Builder: insert labels of chunk c as soon as chunk c is resident. */
    std::thread builder([&]() {
      const size_t elems_per_chunk = chunk_bytes / idx->size_data_per_element_;
      for (size_t c = 0; c < n_chunks; ++c) {
        {
          std::unique_lock<std::mutex> lk(mu);
          cv.wait(lk, [&]() {
            return chunk_done[c] || read_failed.load(std::memory_order_relaxed);
          });
        }
        if (read_failed.load(std::memory_order_relaxed)) return;
        const size_t first = c * elems_per_chunk;
        const size_t last =
            std::min(cur_element_count, first + elems_per_chunk);
        for (size_t i = first; i < last; ++i) {
          idx->label_lookup_[idx->getExternalLabel(i)] = i;
        }
      }
    });

    std::vector<std::thread> readers;
    readers.reserve(n_readers);
    for (unsigned r = 0; r < n_readers; ++r) readers.emplace_back(reader_fn);
    for (auto& t : readers) t.join();
    builder.join();

    if (read_failed.load(std::memory_order_relaxed)) {
      ::close(fd);
      throw std::runtime_error("Index seems to be corrupted or unsupported");
    }
  }

  /* Link-list tail: one bulk read, parsed in memory. */
  {
    size_t got = 0;
    while (got < tail_bytes) {
      const ssize_t n = ::pread(
          fd, tail.get() + got, tail_bytes - got,
          static_cast<off_t>(pos) + static_cast<off_t>(level0_bytes + got));
      if (n <= 0) {
        ::close(fd);
        throw std::runtime_error("Index seems to be corrupted or unsupported");
      }
      got += static_cast<size_t>(n);
    }
  }
  ::close(fd);

  size_t tail_off = 0;
  for (size_t i = 0; i < cur_element_count; i++) {
    unsigned int linkListSize;
    if (tail_off + sizeof(linkListSize) > tail_bytes)
      throw std::runtime_error("Index seems to be corrupted or unsupported");
    std::memcpy(&linkListSize, tail.get() + tail_off, sizeof(linkListSize));
    tail_off += sizeof(linkListSize);
    if (linkListSize == 0) {
      idx->element_levels_[i] = 0;
      idx->linkLists_[i] = nullptr;
    } else {
      if (tail_off + linkListSize > tail_bytes)
        throw std::runtime_error("Index seems to be corrupted or unsupported");
      idx->element_levels_[i] = linkListSize / idx->size_links_per_element_;
      idx->linkLists_[i] = (char*)malloc(linkListSize);
      if (idx->linkLists_[i] == nullptr)
        throw std::runtime_error(
            "Not enough memory: loadIndex failed to allocate linklist");
      std::memcpy(idx->linkLists_[i], tail.get() + tail_off, linkListSize);
      tail_off += linkListSize;
    }
  }
  // the tail must be consumed exactly (stock check: tellg == total_filesize)
  if (tail_off != tail_bytes)
    throw std::runtime_error("Index seems to be corrupted or unsupported");

  for (size_t i = 0; i < cur_element_count; i++) {
    if (idx->isMarkedDeleted(i)) {
      idx->num_deleted_ += 1;
      if (idx->allow_replace_deleted_) idx->deleted_elements.insert(i);
    }
  }
}

static std::priority_queue<std::pair<float, hnswlib::labeltype>>
hnsw_search_with_ef(const hnswlib::HierarchicalNSW<float>* index,
                    const float* query_data, size_t k, size_t ef_search,
                    hnswlib::BaseFilterFunctor* isIdAllowed = nullptr) {
  std::priority_queue<std::pair<float, hnswlib::labeltype>> result;
  if (index == nullptr ||
      index->cur_element_count.load(std::memory_order_relaxed) == 0) {
    return result;
  }

  hnswlib::tableint currObj = index->enterpoint_node_;
  float curdist =
      index->fstdistfunc_(query_data,
                          index->getDataByInternalId(index->enterpoint_node_),
                          index->dist_func_param_);

  for (int level = index->maxlevel_; level > 0; --level) {
    bool changed = true;
    while (changed) {
      changed = false;
      unsigned int *data =
          reinterpret_cast<unsigned int*>(index->get_linklist(currObj, level));
      int size = index->getListCount(data);
      index->metric_hops++;
      index->metric_distance_computations += size;

      hnswlib::tableint *datal =
          reinterpret_cast<hnswlib::tableint*>(data + 1);
      for (int i = 0; i < size; ++i) {
        hnswlib::tableint cand = datal[i];
        if (cand > index->max_elements_) {
          throw std::runtime_error("cand error");
        }
        float d = index->fstdistfunc_(
            query_data, index->getDataByInternalId(cand),
            index->dist_func_param_);

        if (d < curdist) {
          curdist = d;
          currObj = cand;
          changed = true;
        }
      }
    }
  }

  const size_t ef = std::max(ef_search, k);
  const bool bare_bone_search =
      (index->num_deleted_.load(std::memory_order_relaxed) == 0) &&
      (isIdAllowed == nullptr);
  auto top_candidates = bare_bone_search
                            ? index->searchBaseLayerST<true>(
                                  currObj, query_data, ef, isIdAllowed)
                            : index->searchBaseLayerST<false>(
                                  currObj, query_data, ef, isIdAllowed);

  while (top_candidates.size() > k) {
    top_candidates.pop();
  }
  while (!top_candidates.empty()) {
    std::pair<float, hnswlib::tableint> rez = top_candidates.top();
    top_candidates.pop();
    result.emplace(rez.first, index->getExternalLabel(rez.second));
  }
  return result;
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
#pragma omp parallel num_threads(threads)
    {
      log_hnsw_add_omp_once(params_, n);
#pragma omp for schedule(static)
      for (long long i = 0; i < static_cast<long long>(n); ++i) {
        const size_t idx = static_cast<size_t>(i);
        const hnswlib::labeltype label =
            ids ? static_cast<hnswlib::labeltype>(ids[idx])
                : static_cast<hnswlib::labeltype>(current_count());
        active_index()->addPoint(xb + idx * params_.dim, label);
      }
    }
#else
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
      const size_t idx = static_cast<size_t>(i);
      const hnswlib::labeltype label =
          ids ? static_cast<hnswlib::labeltype>(ids[idx])
              : static_cast<hnswlib::labeltype>(current_count());
      active_index()->addPoint(xb + idx * params_.dim, label);
    }
#endif
  }

  void search(size_t nq, const float* xq, size_t k,
              int64_t* out_ids, float* out_distances,
              const VecRuntimeSearchParams* params) const override {
    if (!index_ready()) return;

    const bool prefer_small = !(params_.metric_tag == VEC_M_IP ||
                                params_.metric_tag == VEC_M_COSINE);
    const size_t ef_search =
        (params != nullptr && params->ef_search.has_value() &&
         params->ef_search.value() > 0)
            ? static_cast<size_t>(params->ef_search.value())
            : 0;
    const bool use_custom_ef = (kind_ == IndexKind::Hnsw && ef_search > 0);

    for (size_t qi = 0; qi < nq; ++qi) {
      float* Dq = out_distances + qi * k;
      int64_t* Iq = out_ids + qi * k;

      // Fill defaults in case results < k.
      for (size_t t = 0; t < k; ++t) {
        Dq[t] = prefer_small ? std::numeric_limits<float>::infinity()
                             : -std::numeric_limits<float>::infinity();
        Iq[t] = -1;
      }

      const size_t count = current_count();
      if (count == 0) {
        continue;
      }
      const size_t k_use = std::min(k, count);
      auto result = use_custom_ef
                        ? hnsw_search_with_ef(
                              hnsw_index_.get(), xq + qi * params_.dim, k_use,
                              ef_search)
                        : active_index()->searchKnn(xq + qi * params_.dim, k_use);
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
      // Same file format as the stock loader; see vec_fast_hnsw_load.
      auto idx = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get());
      vec_fast_hnsw_load(idx.get(), space_.get(), path);
      hnsw_index_ = std::move(idx);
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
    return nullptr; // hnswlib only supports HNSW or Flat (brute force).
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
