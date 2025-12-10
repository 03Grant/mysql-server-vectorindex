// vec_index_runtime.h
#pragma once
#include <atomic>
#include <cstddef>
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

struct vid_pk_mapping_t {
  size_t key_length{0};  // length of MySQL-format PK tuple
  std::vector<std::vector<unsigned char>> pk_values;  // indexed by faiss_id
  bool ready{false};

  void clear() {
    key_length = 0;
    pk_values.clear();
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
