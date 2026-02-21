// vec_index_runtime.cc
#include "vec_index_runtime.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <limits>

#include "ut0ut.h"

vec_index_segment_t* vec_index_ctx_t::mutable_segment() {
  if (segments.empty()) {
    return nullptr;
  }
  if (max_vecindex_id == 0) {
    ib::warn() << "VECINDEX: mutable_segment requested with max_vecindex_id=0";
    return nullptr;
  }

  const std::string seg_id = vec_segment_id_from_u32(max_vecindex_id);
  if (seg_id.empty()) {
    ib::warn() << "VECINDEX: mutable_segment failed to derive seg_id for max_vecindex_id="
               << max_vecindex_id;
    return nullptr;
  }

  vec_index_segment_t* seg = vec_find_segment_by_id(this, seg_id);
  if (seg == nullptr) {
    ib::warn() << "VECINDEX: mutable_segment not found for max_vecindex_id="
               << max_vecindex_id;
  }
  return seg;
}

const vec_index_segment_t* vec_index_ctx_t::mutable_segment() const {
  if (segments.empty()) {
    return nullptr;
  }
  if (max_vecindex_id == 0) {
    ib::warn() << "VECINDEX: mutable_segment (const) requested with max_vecindex_id=0";
    return nullptr;
  }

  const std::string seg_id = vec_segment_id_from_u32(max_vecindex_id);
  if (seg_id.empty()) {
    ib::warn() << "VECINDEX: mutable_segment (const) failed to derive seg_id for max_vecindex_id="
               << max_vecindex_id;
    return nullptr;
  }

  auto it = segment_id_map.find(seg_id);
  if (it != segment_id_map.end()) {
    const size_t idx = it->second;
    if (idx < segments.size()) {
      const auto *seg = &segments[idx];
      if (seg->vecindex_id == seg_id) {
        return seg;
      }
    }
  }

  for (size_t i = 0; i < segments.size(); ++i) {
    if (segments[i].vecindex_id == seg_id) {
      return &segments[i];
    }
  }

  ib::warn() << "VECINDEX: mutable_segment (const) not found for max_vecindex_id="
             << max_vecindex_id;
  return nullptr;
}

std::string vec_segment_id_from_u32(uint32_t id) {
  if (id == 0) {
    return {};
  }
  return std::to_string(static_cast<unsigned long long>(id));
}

uint32_t vec_segment_id_to_u32(const std::string& seg_id) {
  if (seg_id.empty()) {
    return 0;
  }
  const char* cstr = seg_id.c_str();
  char* end = nullptr;
  errno = 0;
  unsigned long long value = std::strtoull(cstr, &end, 10);
  if (errno != 0 || end == cstr || *end != '\0') {
    return 0;
  }
  if (value > std::numeric_limits<uint32_t>::max()) {
    return 0;
  }
  return static_cast<uint32_t>(value);
}

void vec_rebuild_segment_id_map(vec_index_ctx_t* ctx) {
  if (ctx == nullptr) {
    return;
  }

  auto &map = ctx->segment_id_map;
  const auto &segments = ctx->segments;

  if (segments.empty()) {
    map.clear();
    return;
  }

  // Incremental refresh: prune stale entries, then add missing ones.
  for (auto it = map.begin(); it != map.end(); ) {
    const size_t idx = it->second;
    if (idx >= segments.size() || segments[idx].vecindex_id != it->first) {
      it = map.erase(it);
    } else {
      ++it;
    }
  }

  for (size_t i = 0; i < segments.size(); ++i) {
    const auto& id = segments[i].vecindex_id;
    if (!id.empty() && map.find(id) == map.end()) {
      map[id] = i;
    }
  }
}

void vec_append_segment_id_map(vec_index_ctx_t* ctx) {
  if (ctx == nullptr) {
    return;
  }

  auto &segments = ctx->segments;
  if (segments.empty()) {
    return;
  }

  const size_t idx = segments.size() - 1;
  const auto &id = segments[idx].vecindex_id;
  if (id.empty()) {
    return;
  }

  auto &map = ctx->segment_id_map;
  auto it = map.find(id);
  if (it == map.end()) {
    map.emplace(id, idx);
    return;
  }
  if (it->second != idx) {
    ib::warn() << "VECINDEX: segment_id_map mismatch on append for seg_id="
               << id << " map_idx=" << it->second << " actual_idx=" << idx
               << "; rebuilding map.";
    vec_rebuild_segment_id_map(ctx);
  }
}

vec_index_segment_t* vec_find_segment_by_id(vec_index_ctx_t* ctx,
                                            const std::string& seg_id) {
  if (ctx == nullptr || seg_id.empty()) {
    return nullptr;
  }

  auto it = ctx->segment_id_map.find(seg_id);
  if (it != ctx->segment_id_map.end()) {
    const size_t idx = it->second;
    if (idx < ctx->segments.size()) {
      auto *seg = &ctx->segments[idx];
      if (seg->vecindex_id == seg_id) {
        return seg;
      }
    }
  }

  for (size_t i = 0; i < ctx->segments.size(); ++i) {
    if (ctx->segments[i].vecindex_id == seg_id) {
      return &ctx->segments[i];
    }
  }

  return nullptr;
}
