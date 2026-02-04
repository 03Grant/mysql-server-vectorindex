// vec_index_runtime.cc
#include "vec_index_runtime.h"

#include <algorithm>

bool vec_pending_delete_add(vec_index_ctx_t* ctx, const std::string& pk_key,
                            uint32_t segment_id, uint64_t vid) {
  if (ctx == nullptr || pk_key.empty()) {
    return false;
  }

  std::unique_lock<std::shared_mutex> lock(ctx->mu);
  for (auto& rec : ctx->pending_deletes) {
    if (rec.pk_key == pk_key) {
      rec.segment_id = segment_id;
      rec.vid = vid;
      rec.status = vec_delete_status_t::PENDING;
      return true;
    }
  }

  vec_pending_delete_t rec;
  rec.pk_key = pk_key;
  rec.segment_id = segment_id;
  rec.vid = vid;
  rec.status = vec_delete_status_t::PENDING;
  ctx->pending_deletes.emplace_back(std::move(rec));
  return true;
}

bool vec_pending_delete_find(vec_index_ctx_t* ctx, const std::string& pk_key,
                             vec_pending_delete_t* out) {
  if (ctx == nullptr || out == nullptr || pk_key.empty()) {
    return false;
  }

  std::shared_lock<std::shared_mutex> lock(ctx->mu);
  for (const auto& rec : ctx->pending_deletes) {
    if (rec.pk_key == pk_key && rec.status == vec_delete_status_t::PENDING) {
      *out = rec;
      return true;
    }
  }

  return false;
}

bool vec_pending_delete_marked(vec_index_ctx_t* ctx, const std::string& pk_key,
                               uint32_t segment_id, uint64_t vid) {
  if (ctx == nullptr || pk_key.empty()) {
    return false;
  }

  std::unique_lock<std::shared_mutex> lock(ctx->mu);
  for (auto& rec : ctx->pending_deletes) {
    if (rec.pk_key == pk_key && rec.segment_id == segment_id &&
        rec.vid == vid) {
      rec.status = vec_delete_status_t::MARKED;
      return true;
    }
  }

  return false;
}

size_t vec_pending_delete_remove_by_pk(vec_index_ctx_t* ctx,
                                       const std::string& pk_key) {
  if (ctx == nullptr || pk_key.empty()) {
    return 0;
  }

  std::unique_lock<std::shared_mutex> lock(ctx->mu);
  auto& vec = ctx->pending_deletes;
  const size_t before = vec.size();
  vec.erase(std::remove_if(vec.begin(), vec.end(),
                           [&](const vec_pending_delete_t& rec) {
                             return rec.pk_key == pk_key;
                           }),
            vec.end());
  return before - vec.size();
}

bool vec_pending_delete_pk_check(const std::string& expected_pk,
                                 const std::string& actual_pk) {
  return expected_pk == actual_pk;
}

vec_index_segment_t* vec_find_segment_by_id(vec_index_ctx_t* ctx,
                                            uint32_t seg_id) {
  if (ctx == nullptr) {
    return nullptr;
  }

  for (auto& seg : ctx->segments) {
    if (seg.vecindex_id == seg_id) {
      return &seg;
    }
  }

  return nullptr;
}
