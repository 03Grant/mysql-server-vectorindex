#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "vec_params.h"

struct dict_index_t;

/* Constants for vec metadata files. */
constexpr uint32_t VEC_META_MAGIC = 0x4D434556;  // "VECM"
constexpr uint16_t VEC_META_VERSION = 1;

enum class VecSegmentState : uint8_t {
  Preparing = 0,
  Committed = 1,
  Tombstone = 2
};

constexpr uint8_t VEC_SEG_FLAG_HAS_PK_MAPPING = 0x01;

#pragma pack(push, 1)
struct VecMetaHeader {
  uint32_t magic;
  uint16_t version;
  uint64_t index_id;
  uint32_t dimension;
  uint8_t  index_type;
  uint8_t  metric_type;
  uint8_t  reserved[12];
};

struct VecSegmentEntry {
  uint64_t seg_id;
  uint64_t count;
  uint8_t  state;
  uint8_t  reserved[7];
  char     file_name[256];
  uint32_t checksum;
};
#pragma pack(pop)

static_assert(sizeof(VecMetaHeader) == 32, "VecMetaHeader size mismatch");
static_assert(sizeof(VecSegmentEntry) == 284, "VecSegmentEntry size mismatch");

/* RAII wrapper for a vec metadata file. */
class VecMetaFile {
 public:
  VecMetaFile() = default;
  ~VecMetaFile();

  VecMetaFile(const VecMetaFile&) = delete;
  VecMetaFile& operator=(const VecMetaFile&) = delete;

  bool open_or_create(const std::string& path, const VecMetaHeader& expected);

  bool append(VecSegmentEntry* entry, long* offset_out);
  bool overwrite(long offset, VecSegmentEntry* entry);

  FILE* handle() const { return fp_; }

 private:
  FILE* fp_{nullptr};
};

/* Helpers for reading a metadata file. */
bool vec_meta_read_all(const std::string& path, VecMetaHeader* header_out,
                       std::vector<VecSegmentEntry>* entries_out);

/* Path helpers. */
bool vec_meta_path_for_index(const dict_index_t* index, std::string* out);
std::string vec_meta_dirname(const std::string& path);
std::string vec_meta_basename(const std::string& path);
std::string vec_meta_join(const std::string& dir, const std::string& file);

/* Utility helpers. */
VecMetaHeader vec_meta_make_header(const dict_index_t* index,
                                   const vec_params_t& params);
uint8_t vec_meta_index_type(const vec_params_t& params);
uint8_t vec_meta_metric_type(const vec_params_t& params);
uint32_t vec_meta_checksum(const VecSegmentEntry& entry);
void vec_meta_fill_entry(VecSegmentEntry* entry, uint64_t seg_id,
                         uint64_t count, VecSegmentState state,
                         const std::string& file_name,
                         uint8_t flags = 0);

inline bool vec_meta_segment_has_pk_mapping(const VecSegmentEntry& entry) {
  return (entry.reserved[0] & VEC_SEG_FLAG_HAS_PK_MAPPING) != 0;
}

inline void vec_meta_mark_pk_mapping(VecSegmentEntry* entry, bool present) {
  if (entry == nullptr) {
    return;
  }
  if (present) {
    entry->reserved[0] |= VEC_SEG_FLAG_HAS_PK_MAPPING;
  } else {
    entry->reserved[0] &= static_cast<uint8_t>(~VEC_SEG_FLAG_HAS_PK_MAPPING);
  }
}
