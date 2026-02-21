// vec_index_runtime.h
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "trx0types.h"
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
static constexpr size_t kVecSegmentIdMaxLen = 256;

struct vid_pk_mapping_t {
  size_t key_length{0};  // length of MySQL-format PK tuple
  std::vector<std::vector<unsigned char>> pk_values;  // indexed by vec_id
  trx_ids_t trx_ids;  // creator transaction ids aligned to vec_id
  bool ready{false};

  void clear() {
    key_length = 0;
    pk_values.clear();
    trx_ids.clear();
    ready = false;
  }

  size_t size() const { return pk_values.size(); }

  const unsigned char *get(size_t idx) const {
    if (!ready || idx >= pk_values.size() || pk_values[idx].empty()) {
      return nullptr;
    }
    return pk_values[idx].data();
  }

  trx_id_t get_trx_id(size_t idx) const {
    if (!ready || idx >= trx_ids.size()) {
      return 0;
    }
    return trx_ids[idx];
  }
};

// Runtime metadata for a single vector segment (mutable or immutable).
struct vec_index_segment_t {
  std::shared_ptr<std::shared_mutex> rw_lock{
      std::make_shared<std::shared_mutex>()};
  std::unique_ptr<IVectorIndex> index;    // concrete vector index
  vid_pk_mapping_t    vid_pk_mapping;     // PK mapping aligned to faiss_ids for this segment
  std::string         index_file_name;    // persisted filename (immutable segments)
  std::string         vecindex_id;        // segment id as string (max 256 bytes)
  bool                immutable{false};   // immutable segments are persisted to disk
};

struct vec_index_ctx_t {
  std::shared_mutex         mu;
  // runtime handler, include one mutable MEM index (seg_id == max_vecindex_id)
  // and several immutable indexes.
  std::vector<vec_index_segment_t> segments;
  // Map segment id -> index in segments. Call vec_rebuild_segment_id_map() after
  // any segment insert/erase or id update. For pure append to segments, call
  // vec_append_segment_id_map(). The rebuild path refreshes entries incrementally.
  std::unordered_map<std::string, size_t> segment_id_map;
  vec_id_allocator_t        id_alloc;       // monotonic id allocator for Faiss IDs
  vec_params_t              params;         // parameters for immutable index not for in_mem_index
  bool                      inited{false};  // inited is true when in_mem_index is created successfully

  // naming helpers used to relate auxiliary tables / persisted index files
  std::string                index_name_prefix;    // base stem for persisted index files
  // shared auxiliary table (PK + seg_id) for all segments
  std::string                aux_table_name;       // resolved aux table name
  dict_table_t*              aux_dict_table{nullptr};  // cached aux dict object

  //SINGLE_AUX: TO BE DELETED
  // when the mutable segment is being built into an immutable one, keep its snapshot here
  vec_index_segment_t        staging_segment;



  bool                       build_in_progress{false};
  std::atomic<bool>          is_rotation_pending{false};
  std::atomic<bool>          needs_aux_refresh{false};  // request user THD to refresh aux cache
  std::atomic<VecBootstrapState> bootstrap_state{VecBootstrapState::NOT_STARTED};
  std::atomic<bool>          bootstrap_load_submitted{false};
  std::atomic<bool>          bootstrap_loaded{false};


  //SINGLE_AUX: TO BE DELETED
  // Precreated aux table waiting to be renamed to _MEM during rotation.
  std::string                pending_aux_name;
  dict_table_t*              pending_aux_dict{nullptr};

  uint32_t                   max_vecindex_id{0};  // MEM seg_id (largest numeric id)

  vec_index_segment_t* mutable_segment();

  const vec_index_segment_t* mutable_segment() const;

  vec_index_segment_t* get_segment(size_t idx) {
    return idx < segments.size() ? &segments[idx] : nullptr;
  }

  const vec_index_segment_t* get_segment(size_t idx) const {
    return idx < segments.size() ? &segments[idx] : nullptr;
  }
};

// Find a segment by vecindex_id (segment id). Returns nullptr if not found.
vec_index_segment_t* vec_find_segment_by_id(vec_index_ctx_t* ctx,
                                            const std::string& seg_id);

std::string vec_segment_id_from_u32(uint32_t id);
uint32_t vec_segment_id_to_u32(const std::string& seg_id);
void vec_rebuild_segment_id_map(vec_index_ctx_t* ctx);
void vec_append_segment_id_map(vec_index_ctx_t* ctx);
