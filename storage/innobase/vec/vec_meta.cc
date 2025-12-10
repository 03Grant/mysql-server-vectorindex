#include "vec_meta.h"

#include <cstring>
#include <fcntl.h>
#include <memory>
#include <sys/stat.h>
#include <utility>
#include <unistd.h>

#include "dict0mem.h"
#include "my_sys.h"
#include "sql/sql_table.h"   // build_table_filename
#include "ut0crc32.h"
#include "ut0ut.h"
#include "univ.i"

namespace {

std::pair<std::string, std::string> split_full_name(const char* full) {
  if (full == nullptr) return {};
  const char* slash = std::strchr(full, '/');
  if (slash == nullptr || slash == full || slash[1] == '\0') return {};
  return {std::string(full, static_cast<size_t>(slash - full)),
          std::string(slash + 1)};
}

inline bool flush_fd(FILE* fp) {
  if (fp == nullptr) return false;
  if (fflush(fp) != 0) return false;
  const int fd = fileno(fp);
  if (fd < 0) return false;
  return fsync(fd) == 0;
}

}  // namespace

VecMetaFile::~VecMetaFile() {
  if (fp_ != nullptr) {
    fclose(fp_);
    fp_ = nullptr;
  }
}

bool VecMetaFile::open_or_create(const std::string& path,
                                 const VecMetaHeader& expected) {
  if (path.empty()) return false;

  if (fp_ != nullptr) {
    fclose(fp_);
    fp_ = nullptr;
  }

  FILE* fp = std::fopen(path.c_str(), "r+b");
  bool created = false;
  if (fp == nullptr) {
    fp = std::fopen(path.c_str(), "w+b");
    created = true;
  }
  if (fp == nullptr) {
    ib::warn() << "VECMETA: failed to open meta file '" << path << "'";
    return false;
  }

  if (created) {
    if (std::fwrite(&expected, sizeof(expected), 1, fp) != 1 ||
        !flush_fd(fp)) {
      ib::warn() << "VECMETA: failed to write header for '" << path << "'";
      std::fclose(fp);
      return false;
    }
    fp_ = fp;
    return true;
  }

  VecMetaHeader on_disk{};
  if (std::fread(&on_disk, sizeof(on_disk), 1, fp) != 1) {
    ib::warn() << "VECMETA: failed to read header for '" << path << "'";
    std::fclose(fp);
    return false;
  }

  if (on_disk.magic != expected.magic || on_disk.version != expected.version ||
      on_disk.index_id != expected.index_id) {
    ib::warn() << "VECMETA: header mismatch for '" << path
               << "', expected index_id=" << expected.index_id;
    std::fclose(fp);
    return false;
  }
  if (on_disk.dimension != expected.dimension ||
      on_disk.index_type != expected.index_type ||
      on_disk.metric_type != expected.metric_type) {
    ib::warn() << "VECMETA: header format mismatch for '" << path
               << "' dim=" << on_disk.dimension
               << " type=" << static_cast<unsigned>(on_disk.index_type)
               << " metric=" << static_cast<unsigned>(on_disk.metric_type);
    std::fclose(fp);
    return false;
  }

  fp_ = fp;
  return true;
}

bool VecMetaFile::append(VecSegmentEntry* entry, long* offset_out) {
  if (fp_ == nullptr || entry == nullptr) return false;

  entry->checksum = vec_meta_checksum(*entry);

  if (std::fseek(fp_, 0, SEEK_END) != 0) return false;
  long off = std::ftell(fp_);
  if (off < 0) return false;

  if (std::fwrite(entry, sizeof(*entry), 1, fp_) != 1 || !flush_fd(fp_)) {
    ib::warn() << "VECMETA: append failed at offset " << off;
    return false;
  }

  if (offset_out != nullptr) {
    *offset_out = off;
  }
  return true;
}

bool VecMetaFile::overwrite(long offset, VecSegmentEntry* entry) {
  if (fp_ == nullptr || entry == nullptr) return false;
  if (offset < 0) return false;

  entry->checksum = vec_meta_checksum(*entry);
  if (std::fseek(fp_, offset, SEEK_SET) != 0) return false;

  if (std::fwrite(entry, sizeof(*entry), 1, fp_) != 1 || !flush_fd(fp_)) {
    ib::warn() << "VECMETA: overwrite failed at offset " << offset;
    return false;
  }
  return true;
}

bool vec_meta_read_all(const std::string& path, VecMetaHeader* header_out,
                       std::vector<VecSegmentEntry>* entries_out) {
  if (header_out == nullptr || entries_out == nullptr || path.empty()) {
    return false;
  }
  entries_out->clear();

  FILE* fp = std::fopen(path.c_str(), "rb");
  if (fp == nullptr) {
    return false;
  }

  auto close_guard = std::unique_ptr<FILE, decltype(&std::fclose)>(
      fp, &std::fclose);

  if (std::fread(header_out, sizeof(*header_out), 1, fp) != 1) {
    ib::warn() << "VECMETA: unable to read header from '" << path << "'";
    return false;
  }

  VecSegmentEntry entry{};
  while (std::fread(&entry, sizeof(entry), 1, fp) == 1) {
    const uint32_t crc = vec_meta_checksum(entry);
    if (crc != entry.checksum) {
      ib::warn() << "VECMETA: checksum mismatch in '" << path
                 << "' seg_id=" << entry.seg_id << " (TODO: crash recovery)";
      break;
    }
    entries_out->push_back(entry);
  }

  return true;
}

bool vec_meta_path_for_index(const dict_index_t* index, std::string* out) {
  if (index == nullptr || index->table == nullptr || out == nullptr) {
    return false;
  }

  const auto parts = split_full_name(index->table->name.m_name);
  if (parts.first.empty() || parts.second.empty()) {
    return false;
  }

  char path[FN_REFLEN + 1]{};
  bool truncated = false;
  size_t len =
      build_table_filename(path, sizeof(path) - 1, parts.first.c_str(),
                           parts.second.c_str(), "", 0, &truncated);
  if (len == 0 || truncated) {
    return false;
  }

  std::string meta(path, len);
  meta.append(".vecindex_")
      .append(std::to_string(
          static_cast<unsigned long long>(index->id)))
      .append(".meta");
  *out = std::move(meta);
  return true;
}

std::string vec_meta_dirname(const std::string& path) {
  const auto pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return {};
  }
  return path.substr(0, pos);
}

std::string vec_meta_basename(const std::string& path) {
  const auto pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return path;
  }
  return path.substr(pos + 1);
}

std::string vec_meta_join(const std::string& dir, const std::string& file) {
  if (file.empty()) return {};
  if (!file.empty() && (file[0] == '/' || file[0] == '\\')) {
    return file;
  }
  if (dir.empty()) {
    return file;
  }
  std::string joined = dir;
  if (joined.back() != '/' && joined.back() != '\\') {
    joined.push_back('/');
  }
  joined.append(file);
  return joined;
}

VecMetaHeader vec_meta_make_header(const dict_index_t* index,
                                   const vec_params_t& params) {
  VecMetaHeader header{};
  header.magic = VEC_META_MAGIC;
  header.version = VEC_META_VERSION;
  header.index_id = index != nullptr ? static_cast<uint64_t>(index->id) : 0;
  header.dimension = params.dim;
  header.index_type = vec_meta_index_type(params);
  header.metric_type = vec_meta_metric_type(params);
  std::memset(header.reserved, 0, sizeof(header.reserved));
  return header;
}

uint8_t vec_meta_index_type(const vec_params_t& params) {
  return params.type_tag;
}

uint8_t vec_meta_metric_type(const vec_params_t& params) {
  return params.metric_tag;
}

uint32_t vec_meta_checksum(const VecSegmentEntry& entry) {
  return ut_crc32(reinterpret_cast<const byte*>(&entry),
                  sizeof(entry) - sizeof(entry.checksum));
}

void vec_meta_fill_entry(VecSegmentEntry* entry, uint64_t seg_id,
                         uint64_t count, VecSegmentState state,
                         const std::string& file_name, uint8_t flags) {
  if (entry == nullptr) return;
  std::memset(entry, 0, sizeof(*entry));
  entry->seg_id = seg_id;
  entry->count = count;
  entry->state = static_cast<uint8_t>(state);
  entry->reserved[0] = flags;

  const std::string base = vec_meta_basename(file_name);
  if (!base.empty()) {
    std::strncpy(entry->file_name, base.c_str(), sizeof(entry->file_name) - 1);
    entry->file_name[sizeof(entry->file_name) - 1] = '\0';
  }
  entry->checksum = vec_meta_checksum(*entry);
}
