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
//
// CONCURRENCY: A segment object is heap-allocated and owned through
// std::shared_ptr (see vec_segment_ptr). Once a segment is part of a published
// version it is never moved or destroyed while any reader still references it
// (the shared_ptr keeps it alive). The per-segment rw_lock guards the in-memory
// index/vid_pk_mapping: appends and the flush index-swap take it exclusively,
// searches take it shared. The `immutable` flag is a one-way latch flipped only
// by the flush writer while holding the context's exclusive lock; the read path
// (vec_search and search-side validation) never reads it, so it carries no
// reader/writer race.
struct vec_index_segment_t {
  std::shared_ptr<std::shared_mutex> rw_lock{
      std::make_shared<std::shared_mutex>()};
  std::unique_ptr<IVectorIndex> index;    // concrete vector index
  vid_pk_mapping_t    vid_pk_mapping;     // PK mapping aligned to faiss_ids for this segment
  std::string         index_file_name;    // persisted filename (immutable segments)
  std::string         vecindex_id;        // segment id as string (max 256 bytes)
  bool                immutable{false};   // immutable segments are persisted to disk
  // Birth segment ids whose entries were re-ingested into this (mutable)
  // segment during crash recovery because their own flush never committed.
  // Their manifest entries are retired (Tombstone) only once this segment's
  // flush commits, i.e. once the adopted entries are durable again.
  std::vector<uint64_t> adopted_seg_ids;
};

// Segments are reference-counted so an immutable version snapshot and the
// writer-side segment list can share the same objects, and so a reader that has
// pinned a version keeps its segments alive even after a concurrent flush/merge
// removes them from the writer-side list.
using vec_segment_ptr = std::shared_ptr<vec_index_segment_t>;

// Immutable snapshot of the segment topology at one instant. A reader pins the
// current version (one atomic load) and reads only that snapshot for the whole
// query, so it observes a complete, internally consistent set of segments and
// never races with a concurrent flush/merge that publishes a new version. A
// published version is never mutated; writers build a fresh one and atomically
// install it (vec_publish_version).
struct vec_index_version_t {
  std::vector<vec_segment_ptr> segments;            // immutable after publish
  std::unordered_map<std::string, size_t> id_map;   // seg_id -> index in segments
  uint32_t max_vecindex_id{0};                       // mutable seg id in this snapshot

  // Returns the segment with the given id, or nullptr. The pointer is valid for
  // as long as the caller keeps this version pinned.
  vec_index_segment_t* find(const std::string& seg_id) const;
  // Returns the mutable segment of this snapshot, or nullptr.
  vec_index_segment_t* mutable_segment() const;
};

using vec_version_ptr = std::shared_ptr<const vec_index_version_t>;

struct vec_index_ctx_t {
  // Writer-serialization lock. Held EXCLUSIVELY by topology writers (flush,
  // merge, drop, bootstrap load) around their mutation of the writer-side state
  // below plus the version publish. Held SHARED only briefly by the insert path
  // so an append cannot land in a segment that flush is sealing. The read path
  // (vec_search) does NOT take this lock; it pins current_version instead.
  std::shared_mutex         mu;
  // Writer-side authoritative segment list. Mutated ONLY while holding `mu`
  // exclusively, and every mutation is followed by vec_commit_topology() before
  // the lock is released so that the published snapshot stays in step. Readers
  // never touch this directly; they read current_version.
  // Includes one mutable MEM index (seg_id == max_vecindex_id) and several
  // immutable indexes.
  std::vector<vec_segment_ptr> segments;
  // Map segment id -> index in segments. Maintained by vec_commit_topology()
  // after any change to `segments` or `max_vecindex_id`.
  std::unordered_map<std::string, size_t> segment_id_map;
  // Immutable published snapshot of {segments, segment_id_map, max_vecindex_id}.
  // Readers pin it with a single atomic load (vec_pin_version); writers replace
  // it with a fresh snapshot (vec_publish_version) while holding `mu`
  // exclusively. Never null once the index is initialized.
  //
  // Accessed exclusively through the std::atomic_load/std::atomic_store free
  // functions on shared_ptr (see vec_pin_version / vec_publish_version). We do
  // not use std::atomic<std::shared_ptr<>> because the toolchain's libstdc++
  // (11) does not implement it; the free functions provide the same
  // atomic, reference-count-safe load/store.
  vec_version_ptr current_version;
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



  std::atomic<bool>          build_in_progress{false};
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
    return idx < segments.size() ? segments[idx].get() : nullptr;
  }

  const vec_index_segment_t* get_segment(size_t idx) const {
    return idx < segments.size() ? segments[idx].get() : nullptr;
  }
};

// Find a segment by vecindex_id (segment id) in the writer-side list. Returns
// nullptr if not found. WRITER-SIDE ONLY: callers must hold ctx->mu. Readers
// must instead pin a version and use vec_index_version_t::find().
vec_index_segment_t* vec_find_segment_by_id(vec_index_ctx_t* ctx,
                                            const std::string& seg_id);

// Like vec_find_segment_by_id but returns a shared owner so the segment stays
// alive after ctx->mu is released (e.g. across a long background build). The
// caller must hold ctx->mu while calling. Returns an empty shared_ptr if the
// segment is not found.
vec_segment_ptr vec_find_segment_shared(vec_index_ctx_t* ctx,
                                        const std::string& seg_id);

std::string vec_segment_id_from_u32(uint32_t id);
uint32_t vec_segment_id_to_u32(const std::string& seg_id);
void vec_rebuild_segment_id_map(vec_index_ctx_t* ctx);
void vec_append_segment_id_map(vec_index_ctx_t* ctx);

// Pin the current published snapshot. One atomic load; never blocks a writer.
// May return nullptr before the index is initialized; callers must check.
vec_version_ptr vec_pin_version(const vec_index_ctx_t* ctx);

// Publish a fresh immutable snapshot from the current writer-side state. MUST be
// called while holding ctx->mu exclusively. Prefer vec_commit_topology(), which
// also refreshes segment_id_map.
void vec_publish_version(vec_index_ctx_t* ctx);

// Rebuild segment_id_map and publish a new version. Call this at the end of
// EVERY writer mutation of ctx->segments / ctx->max_vecindex_id, while still
// holding ctx->mu exclusively. This is the single choke point that keeps the
// published snapshot consistent with the writer-side list.
void vec_commit_topology(vec_index_ctx_t* ctx);
