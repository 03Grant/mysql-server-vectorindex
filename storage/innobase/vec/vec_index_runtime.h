// vec_index_runtime.h
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "vec_id_alloc.h"
#include "vec_params.h"
#include "vec_index.h"

struct dict_table_t;
struct TABLE;
class handler;
class MDL_ticket;

enum class VecBootstrapState : uint8_t {
  NOT_STARTED = 0,
  LOADING = 1,
  READY = 2,
  FAILED = 3,
};

static constexpr char kVecIndexLoadingMsg[] = "Vector index loading, retry";

struct vid_pk_mapping_t {
  size_t key_length{0};  // length of MySQL-format PK tuple
  std::vector<std::vector<unsigned char>> pk_values;  // indexed by faiss_id
  bool ready{false};

  // Sometimes we need to track _MEM index changes. Because everytime we add/remove a vector, we need to update the PK mapping.
  // This is used to record the differences. For example, we delete 2 from {0,1,2,3}
  // When recover, we only reconstruct {0,1,3} so the location changed.
  // At these time, we set mem_diff[3]=1, so we access pk_values[faiss_id=3-1] to get pk.
  bool is_mem_diff{false};
  // mem_diff[faiss_id] = faiss_id - pk_values_index; -1 means missing.
  std::vector<int64_t> mem_diff;

  void clear() {
    key_length = 0;
    pk_values.clear();
    is_mem_diff = false;
    mem_diff.clear();
    ready = false;
  }

  size_t size() const { return pk_values.size(); }

  const unsigned char *get(size_t idx) const {
    if (!ready || idx >= pk_values.size() || pk_values[idx].empty()) {
      return nullptr;
    }
    return pk_values[idx].data();
  }
};

// Bitmap to track deleted vector IDs (VIDs) inside a segment.
struct vecindex_bitmap_t {
  std::vector<uint8_t> bits;

  void clear() { bits.clear(); }

  void ensure_size(size_t nbits) {
    const size_t bytes = (nbits + 7) / 8;
    if (bytes > bits.size()) {
      bits.resize(bytes, 0);
    }
  }

  void mark(size_t idx) {
    const size_t byte = idx / 8;
    const size_t bit = idx % 8;
    ensure_size(idx + 1);
    bits[byte] |= static_cast<uint8_t>(1u << bit);
  }

  bool is_marked(size_t idx) const {
    const size_t byte = idx / 8;
    const size_t bit = idx % 8;
    if (byte >= bits.size()) {
      return false;
    }
    return (bits[byte] >> bit) & 0x01;
  }

  void set(size_t idx, bool value) {
    ensure_size(idx + 1);
    const size_t byte = idx / 8;
    const size_t bit = idx % 8;
    const uint8_t mask = static_cast<uint8_t>(1u << bit);
    if (value) {
      bits[byte] |= mask;
    } else {
      bits[byte] &= static_cast<uint8_t>(~mask);
    }
  }
};

// Runtime metadata for a single vector segment (mutable or immutable).
struct vec_index_segment_t {
  std::unique_ptr<IVectorIndex> index;    // concrete vector index
  dict_table_t       *aux_dict_table{nullptr};  // cached aux dict object bound to this segment
  vid_pk_mapping_t    vid_pk_mapping;     // PK mapping aligned to faiss_ids for this segment
  vecindex_bitmap_t   vecindex_bitmap;    // Deletion bitmap for this segment
  std::string         aux_table_name;     // resolved aux table name for this segment
  std::string         index_file_name;    // persisted filename (immutable segments)
  uint32_t            vecindex_id{0};
  bool                immutable{false};   // immutable segments are persisted to disk
};

struct vec_index_ctx_t {
  std::mutex                mu;
  // runtime handler, include one in_mem_index (segments[0]) and several immutable indexes.
  std::vector<vec_index_segment_t> segments;
  vec_id_allocator_t        id_alloc;       // monotonic id allocator for Faiss IDs
  vec_params_t              params;         // parameters for immutable index not for in_mem_index
  bool                      inited{false};  // inited is true when in_mem_index is created successfully

  // naming helpers used to relate auxiliary tables / persisted index files
  std::string                index_name_prefix;    // base stem for persisted index files

  // when the mutable segment is being built into an immutable one, keep its snapshot here
  vec_index_segment_t        staging_segment;
  bool                       build_in_progress{false};
  std::atomic<bool>          is_rotation_pending{false};
  std::atomic<bool>          needs_aux_refresh{false};  // request user THD to refresh aux cache
  std::atomic<VecBootstrapState> bootstrap_state{VecBootstrapState::NOT_STARTED};
  std::atomic<bool>          bootstrap_load_submitted{false};
  std::atomic<bool>          bootstrap_loaded{false};

  // Precreated aux table waiting to be renamed to _MEM during rotation.
  std::string                pending_aux_name;
  dict_table_t*              pending_aux_dict{nullptr};

  uint32_t                   max_vecindex_id{0};

  vec_index_segment_t* mutable_segment() {
    return segments.empty() ? nullptr : &segments.front();
  }

  const vec_index_segment_t* mutable_segment() const {
    return segments.empty() ? nullptr : &segments.front();
  }

  vec_index_segment_t* get_segment(size_t idx) {
    return idx < segments.size() ? &segments[idx] : nullptr;
  }

  const vec_index_segment_t* get_segment(size_t idx) const {
    return idx < segments.size() ? &segments[idx] : nullptr;
  }
};
