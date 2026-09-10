// vec_tasks.cc
#include "vec_tasks.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <limits>
#include <set>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "trx0trx.h"
#include "ut0ut.h"
#include "my_sys.h"
#include "my_thread.h"
#include "mysqld.h"
#include "current_thd.h"
#include "ha_innodb.h"
#include "my_bitmap.h"
#include "sql/sql_table.h"
#include "sql/sql_base.h"
#include "sql/sql_class.h"
#include "sql/sql_lex.h"
#include "sql/key.h"
#include "sql/mdl.h"
#include "sql/sql_thd_internal_api.h"
#include "sql/transaction.h"
#include "dict0dd.h"
#include "dict0dict.h"
#include "dd/cache/dictionary_client.h"
#include "trx0roll.h"
#include "ut0dbg.h"
#include "vec_aux_tables.h"
#include "vec_faiss_factory.h"
#include "vec_hnswlib_factory.h"
#include "vec_diskann_factory.h"
#include "vec_index.h"
#include "vec_index_runtime.h"
#include "vec_meta.h"
#include "vec_params.h"
#include "row0mysql.h"
#include "row0row.h"
#include "row0sel.h"
#include "data0type.h"
#include "lob0lob.h"
#include "mtr0mtr.h"
#include "rem0rec.h"
#include "vec_txn_buf.h"

namespace {

constexpr uint32_t VEC_SNAPSHOT_MAGIC = 0x56454353;  // "VECS"
constexpr uint16_t VEC_SNAPSHOT_VERSION = 2;

#pragma pack(push, 1)
struct VecSnapshotHeader {
  uint32_t magic;
  uint16_t version;
  uint16_t reserved;
  uint64_t table_id;
  uint64_t index_id;
  uint64_t seg_id;
  uint32_t dim;
  uint64_t total;
  uint64_t pk_count;
};
#pragma pack(pop)

static std::string vec_snapshot_path(const dict_index_t* index,
                                     table_id_t table_id,
                                     space_index_t index_id,
                                     uint32_t seg_id) {
  std::string meta_path;
  if (!vec_meta_path_for_index(index, &meta_path)) {
    return {};
  }
  const std::string dir = vec_meta_dirname(meta_path);
  if (dir.empty()) {
    return {};
  }
  std::string file = "vecsnap_";
  file.append(std::to_string(static_cast<unsigned long long>(table_id)));
  file.push_back('_');
  file.append(std::to_string(static_cast<unsigned long long>(index_id)));
  file.push_back('_');
  file.append(std::to_string(static_cast<unsigned long long>(seg_id)));
  file.append(".tmp");
  return vec_meta_join(dir, file);
}

static bool vec_write_segment_snapshot(dict_index_t* index,
                                       vec_index_ctx_t* ctx,
                                       vec_index_segment_t* seg,
                                       table_id_t table_id,
                                       space_index_t index_id) {
  if (index == nullptr || ctx == nullptr || seg == nullptr ||
      seg->index == nullptr) {
    return false;
  }

  const uint32_t seg_id = vec_segment_id_to_u32(seg->vecindex_id);
  if (seg_id == 0) {
    return false;
  }
  const std::string path =
      vec_snapshot_path(index, table_id, index_id, seg_id);
  if (path.empty()) {
    return false;
  }

  FILE* fp = std::fopen(path.c_str(), "wb");
  if (fp == nullptr) {
    return false;
  }

  auto close_guard = std::unique_ptr<FILE, decltype(&std::fclose)>(
      fp, &std::fclose);

  auto write_bytes = [&](const void* data, size_t len) -> bool {
    return (len == 0) || (std::fwrite(data, 1, len, fp) == len);
  };

  const size_t dim = ctx->params.dim;
  std::shared_lock<std::shared_mutex> seg_lock;
  if (seg->rw_lock) {
    seg_lock = std::shared_lock<std::shared_mutex>(*seg->rw_lock);
  }

  const uint64_t total = seg->index->ntotal();
  const uint64_t pk_count = seg->vid_pk_mapping.pk_values.size();

  VecSnapshotHeader header{};
  header.magic = VEC_SNAPSHOT_MAGIC;
  header.version = VEC_SNAPSHOT_VERSION;
  header.reserved = 0;
  header.table_id = static_cast<uint64_t>(table_id);
  header.index_id = static_cast<uint64_t>(index_id);
  header.seg_id = static_cast<uint64_t>(seg_id);
  header.dim = static_cast<uint32_t>(dim);
  header.total = total;
  header.pk_count = pk_count;

  if (!write_bytes(&header, sizeof(header))) {
    return false;
  }

  for (const auto& pk : seg->vid_pk_mapping.pk_values) {
    const uint32_t len = static_cast<uint32_t>(pk.size());
    if (!write_bytes(&len, sizeof(len))) {
      return false;
    }
    if (!write_bytes(pk.data(), pk.size())) {
      return false;
    }
  }

  if (dim != 0 && total != 0) {
    std::vector<float> buf(dim);
    for (uint64_t i = 0; i < total; ++i) {
      if (!seg->index->reconstruct(static_cast<size_t>(i), buf.data())) {
        return false;
      }
      if (!write_bytes(buf.data(), dim * sizeof(float))) {
        return false;
      }
    }
  }

  if (std::fflush(fp) != 0) {
    return false;
  }
  const int fd = fileno(fp);
  if (fd < 0 || fsync(fd) != 0) {
    return false;
  }
  return true;
}

static bool vec_should_skip_task(const dict_index_t* index) {
  if (index == nullptr || index->table == nullptr) {
    return false;
  }
  if (index->table->to_be_dropped) {
    return true;
  }
  return vec_dict_table_is_aux(index->table);
}

bool vec_extract_pk_columns(
    TABLE *mysql_table, dict_index_t *clust_index, ulint pk_fields,
    const std::function<Field *(size_t)> &field_resolver,
    std::vector<vec_pk_column_t> &out) {
  if (mysql_table == nullptr || clust_index == nullptr ||
      mysql_table->s == nullptr) {
    return false;
  }

  out.clear();
  out.resize(pk_fields);

  for (ulint i = 0; i < pk_fields; ++i) {
    Field *f = field_resolver(static_cast<size_t>(i));
    if (f == nullptr) {
      return false;
    }

    vec_pk_column_t col{};
    col.is_null = f->is_null();

    dict_field_t *df = clust_index->get_field(static_cast<ulint>(i));
    if (df != nullptr && df->col != nullptr) {
      col.mtype = df->col->mtype;
      col.prtype = df->col->prtype;
    }

    if (!col.is_null) {
      const uchar *ptr = f->data_ptr();
      const uint len = f->data_length();
      if (ptr == nullptr && len != 0) {
        return false;
      }

      dtype_t dtype;
      dtype_set(&dtype, col.mtype, col.prtype, df->col->len);
      dfield_t dfield;
      dfield_set_type(&dfield, &dtype);
      std::vector<byte> tmp(df->col->len + 16);
      byte *end = row_mysql_store_col_in_innobase_format(
          &dfield, tmp.data(), true, ptr, df->col->len,
          dict_table_is_comp(clust_index->table));
      const ulint stored_len =
          static_cast<ulint>(end - static_cast<byte *>(tmp.data()));
      col.data.assign(tmp.data(), tmp.data() + stored_len);
    }

    out[i] = std::move(col);
  }

  return true;
}

bool vec_extract_vector_field(TABLE *mysql_table, size_t vec_col_no,
                              size_t dim, std::vector<float> &out) {
  if (mysql_table == nullptr || mysql_table->field == nullptr ||
      vec_col_no >= mysql_table->s->fields || dim == 0) {
    return false;
  }

  Field *vf = mysql_table->field[vec_col_no];
  if (vf == nullptr || vf->is_null()) {
    return false;
  }

  String tmp;
  String *data = vf->val_str(&tmp);
  if (data == nullptr) {
    return false;
  }
  const char *raw = data->ptr();
  const size_t len = data->length();
  const size_t expected = dim * sizeof(float);
  if (raw == nullptr || len != expected) {
    return false;
  }

  out.resize(dim);
  for (size_t i = 0; i < dim; ++i) {
    float value;
    std::memcpy(&value, raw + i * sizeof(float), sizeof(float));
    if (!std::isfinite(value)) {
      return false;
    }
    out[i] = value;
  }

  return true;
}

std::unique_ptr<IVectorIndex> make_mutable_index(const vec_params_t &params) {
  vec_params_t mem_params = params;
  if (mem_params.backend == BackendType::Diskann) {
    mem_params.backend = BackendType::Faiss;
  }
  mem_params.type_tag = VEC_T_FLAT;
  mem_params.size = 0;
  ib::warn() << "VECINDEX: function::make_mutable_index() Before mutable index with backend ";
  switch (mem_params.backend) {
    case BackendType::Faiss:
      return vec_make_faiss_index(mem_params);
    case BackendType::Hnswlib:
      return vec_make_hnswlib_index(mem_params);
    case BackendType::Diskann:
      return vec_make_diskann_index(mem_params);
    default:
      return nullptr;
  }

  ib::warn() << "VECINDEX: function::make_mutable_index() After mutable index with backend ";
}

bool vec_dump_segment(const vec_index_segment_t *seg, size_t dim,
                      std::vector<float> &out, std::vector<int64_t> &ids) {
  ib::warn() << "VECINDEX: function::vec_dump_segment() Dump segment with dim " << dim;
  if (seg == nullptr || seg->index == nullptr || dim == 0) {
    ib::warn() << "VECINDEX: function::vec_dump_segment() Invalid segment or dimension.";
    return false;
  }
  const size_t n = seg->index->ntotal();
  out.resize(n * dim);
  ids.resize(n);

  for (size_t i = 0; i < n; ++i) {
    if (!seg->index->reconstruct(i, out.data() + i * dim)) {
      return false;
    }
    ids[i] = static_cast<int64_t>(i);
  }
  ib::warn() << "VECINDEX: function::vec_dump_segment() Finished dumping segment.";
  return true;
}

bool vec_resolve_index_path(const dict_index_t* index, uint32_t seg_id,
                            std::string *out) {
  if (out == nullptr || index == nullptr || seg_id == 0) {
    return false;
  }
  std::string meta_path;
  if (!vec_meta_path_for_index(index, &meta_path)) {
    return false;
  }
  const std::string dir = vec_meta_dirname(meta_path);
  if (dir.empty()) {
    return false;
  }
  std::string file = "vecseg_";
  file.append(std::to_string(
      static_cast<unsigned long long>(index->table->id)));
  file.push_back('_');
  file.append(std::to_string(
      static_cast<unsigned long long>(index->id)));
  file.push_back('_');
  file.append(std::to_string(
      static_cast<unsigned long long>(seg_id)));
  file.append(".vec");
  *out = vec_meta_join(dir, file);
  return true;
}

bool vec_try_rotate_mem_index(trx_t *trx, dict_index_t *index,
                              vec_index_ctx_t *ctx) {
  ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() 1 Trying to rotate mem index.";
  if (index == nullptr || ctx == nullptr) {
    ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() Invalid arguments.";
    return false;
  }

  ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() 2 Trying to rotate mem index.";
  std::unique_lock<std::shared_mutex> lk(ctx->mu);
  vec_index_segment_t *mutable_seg = ctx->mutable_segment();
  if (mutable_seg == nullptr || mutable_seg->index == nullptr) {
    ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() No mutable segment found.";
    return false;
  }

  ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() 3 Trying to rotate mem index.";
  const std::string prefix =
      ctx->index_name_prefix.empty() ? vec_aux_full_name(index)
                                     : ctx->index_name_prefix;
  if (prefix.empty()) {
    ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() Empty prefix.";
    return false;
  }

  ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() 4 Trying to rotate mem index.";
  if (ctx->max_vecindex_id == 0) {
    ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() max_vecindex_id is 0.";
    return false;
  }
  const uint32_t current_id = ctx->max_vecindex_id;
  const std::string current_id_str = vec_segment_id_from_u32(current_id);
  if (mutable_seg->vecindex_id != current_id_str) {
    ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() "
               << "mutable seg_id mismatch, expected=" << current_id_str
               << " actual=" << mutable_seg->vecindex_id;
    mutable_seg->vecindex_id = current_id_str;
  }
  if (current_id == std::numeric_limits<uint32_t>::max()) {
    ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() "
               << "max_vecindex_id overflow.";
    return false;
  }
  const uint32_t new_mem_id = current_id + 1;

  ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() 5 Creating new mutable index.";
  auto new_index = make_mutable_index(ctx->params);
  if (!new_index) {
    ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() Failed to create mutable index.";
    return false;
  }

  mutable_seg->immutable = true;

  auto fresh = std::make_shared<vec_index_segment_t>();
  fresh->index = std::move(new_index);
  fresh->immutable = false;
  fresh->vecindex_id = vec_segment_id_from_u32(new_mem_id);
  ctx->segments.push_back(std::move(fresh));
  ctx->max_vecindex_id = new_mem_id;
  (void)vec_meta_write_mem_seg_id(index, ctx->params,
                                  static_cast<uint64_t>(new_mem_id));
  // Publish the new topology (old mutable now immutable + fresh mutable) so
  // readers see the switch atomically. Done under the exclusive lock.
  vec_commit_topology(ctx);
  ctx->needs_aux_refresh.store(false);
  lk.unlock();
  ib::warn() << "VECINDEX: function::vec_try_rotate_mem_index() Successfully rotated mem index.";
  return true;
}

class VecBgThdGuard {
 public:
  VecBgThdGuard() = default;
  ~VecBgThdGuard() { cleanup(); }

  bool create() {
    prev_thd_ = current_thd;
    thd_ = create_thd(false, true, true, 0, 0);
    if (thd_ == nullptr) {
      current_thd = prev_thd_;
      return false;
    }
    current_thd = thd_;
    thd_->set_command(COM_DAEMON);
    // DD helpers (Update_dictionary_tables_ctx) require autocommit off.
    thd_->variables.option_bits &= ~OPTION_AUTOCOMMIT;
    thd_->variables.option_bits |= OPTION_NOT_AUTOCOMMIT;
    thd_->variables.transaction_read_only = false;
    thd_->tx_read_only = false;
    thd_->system_thread = SYSTEM_THREAD_BACKGROUND;
    thd_->security_context()->skip_grants();
    // DD helpers invoked below expect a DDL-ish sql_command to be present;
    // set a benign default to avoid debug assertions in transactional DDL paths.
    thd_->lex->sql_command = SQLCOM_CREATE_TABLE;
    return true;
  }

  bool start_trx() {
    if (thd_ == nullptr) {
      return false;
    }
    trx_t *&trx_ref = thd_to_trx(thd_);
    if (trx_ref == nullptr) {
      trx_ref = innobase_trx_allocate(thd_);
      owns_trx_ = true;
    } else {
      owns_trx_ = false;
    }
    trx_ = trx_ref;
    if (trx_ == nullptr) {
      return false;
    }
    trx_start_if_not_started(trx_, true, UT_LOCATION_HERE);
    return true;
  }

  bool commit() {
    if (trx_ == nullptr || thd_ == nullptr) {
      return false;
    }
    if (trx_commit_for_mysql(trx_) != DB_SUCCESS) {
      trans_rollback(thd_);
      return false;
    }
    if (trans_commit_stmt(thd_, false)) {
      trans_rollback(thd_);
      return false;
    }
    trans_commit(thd_);
    return true;
  }

  void rollback() {
    if (trx_ != nullptr) {
      trx_rollback_to_savepoint(trx_, nullptr);
    }
    if (thd_ != nullptr) {
      trans_rollback_stmt(thd_);
      trans_rollback(thd_);
    }
  }

  trx_t *trx() const { return trx_; }
  THD *thd() const { return thd_; }

 private:
  void cleanup() {
    if (thd_ != nullptr) {
      current_thd = thd_;
      if (trx_t *trx = thd_to_trx(thd_); trx != nullptr) {
        // Ensure any active trx is fully rolled back before THD teardown.
        // innobase_close_connection() may skip rollback for read-only trx,
        // which can trigger assertions in trx_free_for_background().
        (void)trx_rollback_for_mysql(trx);
      }
      destroy_thd(thd_);
      thd_ = nullptr;
      current_thd = prev_thd_;
    }
    trx_ = nullptr;
    owns_trx_ = false;
  }

  THD *thd_{nullptr};
  trx_t *trx_{nullptr};
  THD *prev_thd_{nullptr};
  bool owns_trx_{false};
};

/* Flush a finished segment artifact to stable storage before its Committed
manifest record is written. Best effort: an unreadable path is skipped (the
aux-table recovery path remains the safety net). */
static void vec_fsync_path(const std::string &path) {
  if (path.empty()) {
    return;
  }
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    return;
  }
  (void)::fsync(fd);
  (void)::close(fd);
}

bool vec_bg_build_task(dict_index_t *index, vec_index_ctx_t *ctx,
                       vec_index_segment_t *seg) {
  ib::warn() << "VECINDEX: function::vec_bg_build_task() Building index segment in bg.";
  if (index == nullptr || ctx == nullptr || seg == nullptr) {
    ib::warn() << "VECINDEX: function::vec_bg_build_task() Invalid arguments.";
    return false;
  }
  if (vec_should_skip_task(index)) {
    ib::warn() << "VECINDEX: function::vec_bg_build_task() Skip build for dropping/aux table.";
    return false;
  }
  ctx->build_in_progress = true;
  auto clear_build_flag = [&]() {
    ctx->build_in_progress = false;
  };

  std::vector<float> xb;
  std::vector<int64_t> ids;
  const size_t dim = ctx->params.dim;
  size_t total = 0;
  {
    std::shared_lock<std::shared_mutex> seg_lock;
    if (seg->rw_lock) {
      seg_lock = std::shared_lock<std::shared_mutex>(*seg->rw_lock);
    }
    if (dim == 0 || seg->index == nullptr) {
      clear_build_flag();
      ib::warn() << "VECINDEX: function::vec_bg_build_task() Invalid segment data.";
      return false;
    }
    total = seg->index->ntotal();

    const auto &pk_values = seg->vid_pk_mapping.pk_values;
    for (size_t i = 0; i < total; ++i) {
      if (i >= pk_values.size() || pk_values[i].empty()) {
        clear_build_flag();
        ib::warn() << "VECINDEX: missing pk mapping for seg_id="
                   << seg->vecindex_id << " vid=" << i;
        return false;
      }
    }

    xb.resize(total * dim);
    ids.resize(total);
    for (size_t i = 0; i < total; ++i) {
      if (!seg->index->reconstruct(i, xb.data() + i * dim)) {
        clear_build_flag();
        ib::warn() << "VECINDEX: failed to reconstruct seg_id="
                   << seg->vecindex_id << " vid=" << i;
        return false;
      }
      ids[i] = static_cast<int64_t>(i);
    }
  }

  const uint32_t seg_id = vec_segment_id_to_u32(seg->vecindex_id);
  if (seg_id == 0) {
    clear_build_flag();
    ib::warn() << "VECINDEX: function::vec_bg_build_task() Invalid seg_id.";
    return false;
  }

  std::string index_path;
  if (!vec_resolve_index_path(index, seg_id, &index_path)) {
    clear_build_flag();
    ib::warn() << "VECINDEX: function::vec_bg_build_task() Failed to resolve index path.";
    return false;
  }

  const std::string index_basename = vec_meta_basename(index_path);
  auto log_event = [&](VecSegmentState state, bool has_pk_mapping) {
    if (!vec_meta_append_event(index, ctx->params,
                               static_cast<uint64_t>(seg_id),
                               static_cast<uint64_t>(total),
                               state, index_basename, has_pk_mapping)) {
      ib::warn() << "VECMETA: failed to append event state="
                 << static_cast<unsigned>(state) << " seg_id="
                 << seg->vecindex_id;
    }
  };
  log_event(VecSegmentState::Preparing, false);

  std::unique_ptr<IVectorIndex> target;
  switch (ctx->params.backend) {
    case BackendType::Faiss:
      target = vec_make_faiss_index(ctx->params);
      break;
    case BackendType::Hnswlib:
      target = vec_make_hnswlib_index(ctx->params);
      break;
    case BackendType::Diskann:
      target = vec_make_diskann_index(ctx->params);
      break;
    default:
      break;
  }
  if (!target) {
    clear_build_flag();
    ib::warn() << "VECINDEX: function::vec_bg_build_task() Failed to create target index.";
    return false;
  }

  if (!ids.empty()) {
    target->train(ids.size(), xb.data());
    target->add(ids.size(), xb.data(), ids.data());
  }

  target->save(index_path);
  if (ctx->params.backend == BackendType::Diskann) {
    target->load(index_path);
  }
  log_event(VecSegmentState::BuiltIndex, false);
  const std::string mapping_path = vec_vid_pk_mapping_path(index_path);
  bool pk_mapping_saved = false;
  if (!mapping_path.empty()) {
    pk_mapping_saved =
        vec_vid_pk_mapping_save(seg->vid_pk_mapping, mapping_path);
    if (!pk_mapping_saved) {
      ib::warn() << "VECINDEX: failed to persist PK mapping to '"
                 << mapping_path << "'";
    } else {
      log_event(VecSegmentState::PkmapSaved, true);
    }
  }

  {
    std::unique_lock<std::shared_mutex> seg_lock;
    if (seg->rw_lock) {
      seg_lock = std::unique_lock<std::shared_mutex>(*seg->rw_lock);
    }
    seg->index_file_name = index_path;
    if (target) {
      seg->index = std::move(target);
    }
  }

  /* The Committed manifest record is the flush's commit point: recovery
  trusts a Committed segment's files, so they must be durable first. */
  vec_fsync_path(index_path);
  if (pk_mapping_saved) {
    vec_fsync_path(mapping_path);
  }

  log_event(VecSegmentState::Committed, pk_mapping_saved);

  /* Entries adopted from segments lost in a previous crash are durable in
  this committed segment now; retire their manifest entries so future
  recoveries stop re-ingesting them. Ordered after the Committed record: a
  crash in between merely re-adopts rows that also exist here, which the
  query-time PK deduplication tolerates. */
  std::vector<uint64_t> adopted;
  {
    std::unique_lock<std::shared_mutex> seg_lock;
    if (seg->rw_lock) {
      seg_lock = std::unique_lock<std::shared_mutex>(*seg->rw_lock);
    }
    adopted.swap(seg->adopted_seg_ids);
  }
  for (const uint64_t adopted_id : adopted) {
    if (!vec_meta_append_event(index, ctx->params, adopted_id, 0,
                               VecSegmentState::Tombstone, "", false)) {
      ib::warn() << "VECMETA: failed to append Tombstone for adopted seg_id="
                 << adopted_id;
    }
  }

  clear_build_flag();
  ib::warn() << "VECINDEX: function::vec_bg_build_task() Finished building index segment.";
  return true;
}

static dberr_t vec_complete_direct_diskann_build_impl(
    dict_index_t *index, const std::string &data_path, uint64_t row_count,
    const vid_pk_mapping_t &mapping) {
  if (index == nullptr || index->vec_runtime == nullptr ||
      index->vec_params == nullptr || row_count == 0) {
    return DB_SUCCESS;
  }

  vec_index_ctx_t *ctx = index->vec_runtime;
  if (ctx->params.backend != BackendType::Diskann) {
    ib::warn() << "VECINDEX: direct DiskANN DDL build called for non-DiskANN "
               << "index '" << (index->name ? index->name : "(null)") << "'";
    return DB_ERROR;
  }

  const uint32_t seg_id = ctx->max_vecindex_id;
  if (seg_id == 0 || seg_id == std::numeric_limits<uint32_t>::max()) {
    return DB_ERROR;
  }

  std::string index_path;
  if (!vec_resolve_index_path(index, seg_id, &index_path)) {
    return DB_ERROR;
  }

  std::string build_error;
  if (!vec_diskann_build_from_fbin(ctx->params,
                                   static_cast<size_t>(row_count), data_path,
                                   index_path, &build_error)) {
    ib::warn() << "VECINDEX: direct DiskANN DDL build failed for index '"
               << (index->name ? index->name : "(null)")
               << "' path='" << index_path << "' error=" << build_error;
    return DB_ERROR;
  }

  const std::string mapping_path = vec_vid_pk_mapping_path(index_path);
  if (mapping_path.empty() ||
      !vec_vid_pk_mapping_save(mapping, mapping_path)) {
    ib::warn() << "VECINDEX: failed to persist direct DiskANN PK mapping to '"
               << mapping_path << "'";
    vec_diskann_remove_artifacts(index_path);
    std::remove(mapping_path.c_str());
    return DB_ERROR;
  }

  std::unique_ptr<IVectorIndex> immutable =
      vec_make_diskann_index(ctx->params);
  if (!immutable) {
    vec_diskann_remove_artifacts(index_path);
    std::remove(mapping_path.c_str());
    return DB_ERROR;
  }

  try {
    immutable->load(index_path);
  } catch (const std::exception &e) {
    ib::warn() << "VECINDEX: failed to load direct DiskANN immutable segment '"
               << index_path << "' error=" << e.what();
    vec_diskann_remove_artifacts(index_path);
    std::remove(mapping_path.c_str());
    return DB_ERROR;
  }

  auto new_mutable = make_mutable_index(ctx->params);
  if (!new_mutable) {
    vec_diskann_remove_artifacts(index_path);
    std::remove(mapping_path.c_str());
    return DB_ERROR;
  }

  const std::string seg_id_str = vec_segment_id_from_u32(seg_id);
  const uint32_t new_mem_id = seg_id + 1;
  const std::string new_mem_id_str = vec_segment_id_from_u32(new_mem_id);

  if (!vec_meta_append_event(index, ctx->params, seg_id, row_count,
                             VecSegmentState::BuiltIndex,
                             vec_meta_basename(index_path), false)) {
    ib::warn() << "VECMETA: failed to append BuiltIndex for direct DiskANN "
               << "DDL build seg_id=" << seg_id;
  }
  if (!vec_meta_append_event(index, ctx->params, seg_id, row_count,
                             VecSegmentState::PkmapSaved,
                             vec_meta_basename(index_path), true)) {
    ib::warn() << "VECMETA: failed to append PkmapSaved for direct DiskANN "
               << "DDL build seg_id=" << seg_id;
  }
  if (!vec_meta_append_event(index, ctx->params, seg_id, row_count,
                             VecSegmentState::Committed,
                             vec_meta_basename(index_path), true)) {
    ib::warn() << "VECMETA: failed to append Committed for direct DiskANN "
               << "DDL build seg_id=" << seg_id;
    vec_diskann_remove_artifacts(index_path);
    std::remove(mapping_path.c_str());
    return DB_ERROR;
  }

  {
    std::lock_guard<std::shared_mutex> ctx_lock(ctx->mu);
    vec_index_segment_t *mutable_seg = ctx->mutable_segment();
    if (mutable_seg == nullptr || mutable_seg->vecindex_id != seg_id_str ||
        mutable_seg->rw_lock == nullptr) {
      vec_diskann_remove_artifacts(index_path);
      std::remove(mapping_path.c_str());
      return DB_ERROR;
    }

    {
      std::unique_lock<std::shared_mutex> seg_lock(*mutable_seg->rw_lock);
      mutable_seg->index = std::move(immutable);
      mutable_seg->vid_pk_mapping.clear();
      mutable_seg->index_file_name = index_path;
      mutable_seg->immutable = true;
    }

    auto fresh = std::make_shared<vec_index_segment_t>();
    fresh->index = std::move(new_mutable);
    fresh->immutable = false;
    fresh->vecindex_id = new_mem_id_str;
    ctx->segments.push_back(std::move(fresh));
    ctx->max_vecindex_id = new_mem_id;
    vec_commit_topology(ctx);
  }

  if (!vec_meta_write_mem_seg_id(index, ctx->params,
                                 static_cast<uint64_t>(new_mem_id))) {
    return DB_ERROR;
  }

  ctx->bootstrap_state.store(VecBootstrapState::READY,
                             std::memory_order_release);
  ctx->bootstrap_loaded.store(true, std::memory_order_release);
  ctx->bootstrap_load_submitted.store(true, std::memory_order_release);
  ctx->is_rotation_pending.store(false, std::memory_order_release);
  ctx->build_in_progress = false;
  ctx->needs_aux_refresh.store(false, std::memory_order_release);
  ib::warn() << "VECINDEX: direct DiskANN DDL build completed for index '"
             << (index->name ? index->name : "(null)")
             << "' seg_id=" << seg_id << " rows=" << row_count
             << " file=" << vec_meta_basename(index_path);
  return DB_SUCCESS;
}

bool vec_rotate_mem_index_bg(dict_index_t *index) {
  ib::warn() << "VECINDEX: function::vec_rotate_mem_index_bg() Rotating mem index in bg.";
  bool rotated_ok = false;
  bool bg_built = false;

  struct PendingGuard {
    vec_index_ctx_t* ctx{nullptr};
    ~PendingGuard() {
      if (ctx != nullptr) {
        ctx->is_rotation_pending.store(false);
      }
    }
  } pending_guard{};

  if (index == nullptr || index->table == nullptr || index->vec_runtime == nullptr) {
    return false;
  }
  vec_index_ctx_t* ctx = index->vec_runtime;
  pending_guard.ctx = ctx;

  if (vec_should_skip_task(index)) {
    ib::warn() << "VECINDEX: function::vec_rotate_mem_index_bg() Skip rotation for dropping/aux table.";
    return false;
  }

  uint32_t old_mem_id = 0;
  {
    std::shared_lock<std::shared_mutex> lk(ctx->mu);
    old_mem_id = ctx->max_vecindex_id;
  }

  rotated_ok = vec_try_rotate_mem_index(nullptr, index, ctx);
  if (rotated_ok) {
    const std::string old_mem_id_str = vec_segment_id_from_u32(old_mem_id);
    // Pin a shared owner of the just-sealed segment. Holding the shared_ptr
    // keeps the segment object alive across the (unlocked) snapshot write and
    // background build even if a concurrent merge removes it from the
    // writer-side list in the meantime.
    vec_segment_ptr staging_sp;
    {
      std::shared_lock<std::shared_mutex> lk(ctx->mu);
      staging_sp = vec_find_segment_shared(ctx, old_mem_id_str);
    }
    vec_index_segment_t* staging_seg = staging_sp.get();
    if (staging_seg != nullptr && staging_seg->immutable) {
      if (!vec_write_segment_snapshot(index, ctx, staging_seg, index->table->id,
                                      index->id)) {
        ib::warn() << "VECMETA: failed to write segment snapshot for index "
                   << (index->name ? index->name : "(null)")
                   << " seg_id=" << staging_seg->vecindex_id;
      }
      bg_built = vec_bg_build_task(index, ctx, staging_seg);
    }
  }

  const bool ok = rotated_ok && bg_built;
  ib::warn() << "VECINDEX: function::vec_rotate_mem_index_bg() Rotation "
             << (ok ? "succeeded." : "failed.");
  return ok;
}

}  // namespace

dberr_t vec_complete_direct_diskann_build(dict_index_t *index,
                                          const std::string &data_path,
                                          uint64_t row_count,
                                          const vid_pk_mapping_t &mapping) {
  return vec_complete_direct_diskann_build_impl(index, data_path, row_count,
                                                mapping);
}

bool vec_load_aux_cache_for_segment(dict_index_t *vec_index,
                                    vec_index_ctx_t *ctx,
                                    vec_index_segment_t *seg, THD *thd) {
  if (vec_index == nullptr || vec_index->table == nullptr || ctx == nullptr ||
      seg == nullptr || thd == nullptr) {
    return false;
  }

  if (seg->vid_pk_mapping.ready) {
    // Usually this way
    return true;
  }

  std::unique_lock<std::shared_mutex> seg_lock;
  if (seg->rw_lock) {
    seg_lock = std::unique_lock<std::shared_mutex>(*seg->rw_lock);
  }
  if (seg->vid_pk_mapping.ready) {
    return true;
  }

  std::string aux_name = ctx->aux_table_name;
  if (aux_name.empty()) {
    aux_name = vec_aux_active_name(vec_index);
    if (aux_name.empty()) {
      return false;
    }
  }


  const std::string mapping_path =
      vec_vid_pk_mapping_path(seg->index_file_name);
  if (!mapping_path.empty() &&
      vec_vid_pk_mapping_load(mapping_path, &seg->vid_pk_mapping)) {
    ib::warn() << "VECINDEX: loaded PK mapping file '" << mapping_path
               << "' rows=" << seg->vid_pk_mapping.size();
    seg->vid_pk_mapping.ready = true;
    return true;
  }
  ib::warn() << "VECINDEX: aux table stores seg_id only; skip aux cache rebuild"
             << " for seg_id=" << seg->vecindex_id;
  return true;
}

bool vec_rotate_mem_index(trx_t *trx, dict_index_t *index) {
  ib::warn() << "VECINDEX: function::vec_rotate_mem_index() Rotating mem index.";
  if (trx == nullptr || index == nullptr || index->vec_runtime == nullptr) {
    ib::warn() << "VECINDEX: function::vec_rotate_mem_index() Invalid arguments.";
    return false;
  }

  /* Ensure caller is driving from a user thread; rotation performs DDL. */
  if (trx->state.load(std::memory_order_acquire) != TRX_STATE_ACTIVE) {
    ib::warn() << "VECINDEX: function::vec_rotate_mem_index() Transaction not active.";
    return false;
  }

  return vec_try_rotate_mem_index(trx, index, index->vec_runtime);
}

VecTaskManager &VecTaskManager::instance() {
  static VecTaskManager instance;
  return instance;
}

VecTaskManager::~VecTaskManager() { stop(); }

void VecTaskManager::ensure_started() {
  if (!running.load(std::memory_order_acquire)) {
    start();
  }
}

void VecTaskManager::start() {
  bool expected = false;
  if (!running.compare_exchange_strong(expected, true,
                                       std::memory_order_acq_rel)) {
    return;
  }
  worker_thread = std::thread(&VecTaskManager::worker_loop, this);
}

void VecTaskManager::stop() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    if (!running.load(std::memory_order_acquire)) {
      return;
    }
    running.store(false, std::memory_order_release);
  }
  cv.notify_all();
  if (worker_thread.joinable()) {
    worker_thread.join();
  }
}

void VecTaskManager::submit_task(dict_index_t *index) {
  if (index == nullptr || index->table == nullptr) {
    return;
  }
  ensure_started();

  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    const uint64_t key = static_cast<uint64_t>(index->id);
    if (pending_index_ids.find(key) != pending_index_ids.end()) {
      return;
    }
    tasks.push({index, index->id});
    pending_index_ids.insert(key);
  }
  ib::warn() << "VECINDEX: function::VecTaskManager::submit_task() Task submitted for index ID ";
  cv.notify_one();
}

void VecTaskManager::worker_loop() {
  bool thread_initialized = !my_thread_init();
  struct ThreadGuard {
    bool ok{false};
    ~ThreadGuard() {
      if (ok) {
        my_thread_end();
      }
    }
  } thread_guard{thread_initialized};

  if (!thread_initialized) {
    running.store(false, std::memory_order_release);
    cv.notify_all();
    return;
  }

  while (true) {
    VecBuildTask task{};
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      cv.wait(lock, [this] {
        return !tasks.empty() || !running.load(std::memory_order_acquire);
      });

      if (!running.load(std::memory_order_acquire) && tasks.empty()) {
        return;
      }

      task = tasks.front();
      tasks.pop();
      pending_index_ids.erase(static_cast<uint64_t>(task.index_id));
    }

    process_task(task);
  }
}

void VecTaskManager::process_task(VecBuildTask task) {
  if (task.index == nullptr || task.index_id == 0) {
    return;
  }
  vec_rotate_mem_index_bg(task.index);
}

/** Rebuild the mutable segment from the auxiliary table after a restart.

Recovers not only the rows born in the persisted mutable segment but also the
rows of every "lost" birth segment: a segment id below the mutable id whose
latest manifest state is neither Committed (its flush never completed, so the
segment file on disk, if any, must be ignored) nor Tombstone (its entries were
already carried into a committed successor by a flush or merge). Such rows
survive only in the auxiliary table, and are re-ingested here into the mutable
segment; the lost ids are recorded as adopted_seg_ids and tombstoned once the
adopting segment's own flush commits.

@param covered_seg_ids birth ids whose latest manifest state is Committed or
Tombstone (their entries are durable elsewhere).
@param manifest_lost_ids non-covered ids that appear in the manifest; adopted
even when they contribute no rows, so their stale entries get retired. */
bool vec_recover_mutable_mem_index(dict_index_t *index, vec_index_ctx_t *ctx,
                                   THD *thd,
                                   const std::set<uint32_t> &covered_seg_ids,
                                   const std::vector<uint64_t> &manifest_lost_ids) {
  ib::warn() << "VECMETA: attempting mutable MEM recovery for index "
             << (index != nullptr && index->name ? index->name : "(null)");

  if (index == nullptr || ctx == nullptr || thd == nullptr ||
      index->table == nullptr) {
    return false;
  }

  vec_index_segment_t *seg = nullptr;
  for (auto &candidate : ctx->segments) {
    if (candidate && !candidate->immutable) {
      seg = candidate.get();
      break;
    }
  }
  if (seg == nullptr || seg->index == nullptr) {
    return false;
  }
  uint64_t persisted_id = 0;
  if (!vec_meta_read_mem_seg_id(index, ctx->params, &persisted_id) ||
      persisted_id == 0 ||
      persisted_id > std::numeric_limits<uint32_t>::max()) {
    ib::warn() << "VECMETA: failed to read MEM seg_id from meta log for index "
               << (index->name ? index->name : "(null)");
    return false;
  }
  const uint32_t seg_id = static_cast<uint32_t>(persisted_id);
  if (seg_id < ctx->max_vecindex_id) {
    ib::warn() << "VECMETA: MEM seg_id=" << seg_id
               << " is smaller than max immutable seg_id="
               << ctx->max_vecindex_id;
    return false;
  }
  seg->vecindex_id = vec_segment_id_from_u32(seg_id);
  ctx->max_vecindex_id = seg_id;
  vec_commit_topology(ctx);

  if (seg->index->ntotal() != 0) {
    ib::warn() << "VECMETA: mutable index already populated, skipping MEM "
               << "recovery for index "
               << (index->name ? index->name : "(null)");
    return true;
  }

  const std::string prefix =
      ctx->index_name_prefix.empty() ? vec_aux_full_name(index)
                                     : ctx->index_name_prefix;
  if (prefix.empty()) {
    return false;
  }

  std::string mem_name = ctx->aux_table_name;
  if (mem_name.empty()) {
    mem_name = prefix;
  }
  if (!vec_aux_table_exists(mem_name)) {
    ib::warn() << "VECMETA: MEM table '" << mem_name
               << "' not found, skip mutable recovery.";
    return true;
  }

  if (ctx->aux_table_name.empty()) {
    ctx->aux_table_name = mem_name;
  }

  dict_index_t *clust_index = index->table->first_index();
  if (clust_index == nullptr) {
    return false;
  }

  const ulint pk_fields = dict_index_get_n_unique(clust_index);
  if (pk_fields == 0) {
    return false;
  }

  const dict_field_t *vec_field = index->get_field(0);
  if (vec_field == nullptr || vec_field->col == nullptr) {
    return false;
  }

  const size_t vec_col_no = dict_col_get_no(vec_field->col);
  const size_t dim = ctx->params.dim;
  if (dim == 0) {
    return false;
  }

  auto split_name = [](const std::string &full, std::string *db,
                       std::string *tbl) -> bool {
    if (full.empty() || db == nullptr || tbl == nullptr) {
      return false;
    }
    const auto slash = full.find('/');
    if (slash == std::string::npos || slash == 0 || slash + 1 >= full.size()) {
      return false;
    }
    *db = full.substr(0, slash);
    *tbl = full.substr(slash + 1);
    return !(db->empty() || tbl->empty());
  };

  std::string aux_db, aux_tbl;
  if (!split_name(mem_name, &aux_db, &aux_tbl)) {
    return false;
  }

  std::string base_db, base_tbl;
  if (index->table->name.m_name == nullptr ||
      !split_name(index->table->name.m_name, &base_db, &base_tbl)) {
    return false;
  }

  MDL_request aux_mdl;
  MDL_request base_mdl;
  MDL_REQUEST_INIT(&aux_mdl, MDL_key::TABLE, aux_db.c_str(), aux_tbl.c_str(),
                   MDL_SHARED_READ, MDL_EXPLICIT);
  MDL_REQUEST_INIT(&base_mdl, MDL_key::TABLE, base_db.c_str(),
                   base_tbl.c_str(), MDL_SHARED_READ, MDL_EXPLICIT);

  if (thd->mdl_context.acquire_lock(&aux_mdl, 0)) {
    ib::warn() << "VECMETA: failed to acquire MDL on MEM table '" << mem_name
               << "'";
    return false;
  }

  if (thd->mdl_context.acquire_lock(&base_mdl, 0)) {
    ib::warn() << "VECMETA: failed to acquire MDL on base table '"
               << index->table->name.m_name << "'";
    thd->mdl_context.release_lock(aux_mdl.ticket);
    return false;
  }

  struct MdlGuard {
    THD *thd{nullptr};
    MDL_request *req{nullptr};
    ~MdlGuard() {
      if (thd != nullptr && req != nullptr && req->ticket != nullptr) {
        thd->mdl_context.release_lock(req->ticket);
        req->ticket = nullptr;
      }
    }
  };

  MdlGuard aux_guard{thd, &aux_mdl};
  MdlGuard base_guard{thd, &base_mdl};

  dd::cache::Dictionary_client *client = dd::get_dd_client(thd);
  dd::cache::Dictionary_client::Auto_releaser releaser(client);

  const dd::Table *dd_aux = nullptr;
  const dd::Table *dd_base = nullptr;
  if (client->acquire<dd::Table>(aux_db.c_str(), aux_tbl.c_str(), &dd_aux) ||
      dd_aux == nullptr) {
    ib::warn() << "VECMETA: failed to acquire DD for MEM table '" << mem_name
               << "'";
    return false;
  }

  if (client->acquire<dd::Table>(base_db.c_str(), base_tbl.c_str(), &dd_base) ||
      dd_base == nullptr) {
    ib::warn() << "VECMETA: failed to acquire DD for base table '"
               << index->table->name.m_name << "'";
    return false;
  }

  char aux_path[FN_REFLEN + 1]{};
  char base_path[FN_REFLEN + 1]{};
  bool aux_trunc = false;
  bool base_trunc = false;
  size_t aux_path_len =
      build_table_filename(aux_path, sizeof(aux_path) - 1 - reg_ext_length,
                           aux_db.c_str(), aux_tbl.c_str(), "", 0, &aux_trunc);
  size_t base_path_len = build_table_filename(
      base_path, sizeof(base_path) - 1 - reg_ext_length, base_db.c_str(),
      base_tbl.c_str(), "", 0, &base_trunc);
  if (aux_path_len == 0 || aux_trunc || base_path_len == 0 || base_trunc) {
    ib::warn() << "VECMETA: failed to build table path for MEM or base table "
               << "during recovery.";
    return false;
  }

  TABLE *aux_table =
      open_table_uncached(thd, aux_path, aux_db.c_str(), aux_tbl.c_str(),
                          false, true, *dd_aux);
  TABLE *base_table =
      open_table_uncached(thd, base_path, base_db.c_str(), base_tbl.c_str(),
                          false, true, *dd_base);

  if (aux_table == nullptr || aux_table->file == nullptr || base_table == nullptr ||
      base_table->file == nullptr) {
    ib::warn() << "VECMETA: failed to open MEM or base table for recovery.";
    if (aux_table != nullptr) {
      intern_close_table(aux_table);
    }
    if (base_table != nullptr) {
      intern_close_table(base_table);
    }
    return false;
  }

  struct TableGuard {
    TABLE *t{nullptr};
    ~TableGuard() {
      if (t != nullptr) {
        intern_close_table(t);
        t = nullptr;
      }
    }
  };

  TableGuard aux_table_guard{aux_table};
  TableGuard base_table_guard{base_table};

  trx_t *trx = thd_to_trx(thd);
  bool allocated_trx = false;
  if (trx == nullptr) {
    trx = innobase_trx_allocate(thd);
    if (trx == nullptr) {
      ib::warn() << "VECMETA: failed to allocate trx for MEM recovery";
      return false;
    }
    thd_to_trx(thd) = trx;
    allocated_trx = true;
  }
  trx_start_if_not_started(trx, true, UT_LOCATION_HERE);

  struct TrxCleanupGuard {
    THD *thd{nullptr};
    trx_t *trx{nullptr};
    bool owned{false};
    bool committed{false};
    ~TrxCleanupGuard() {
      if (trx == nullptr) {
        return;
      }
      if (!committed && owned &&
          trx->state.load(std::memory_order_relaxed) != TRX_STATE_NOT_STARTED) {
        trx_rollback_to_savepoint(trx, nullptr);
      }
      if (owned) {
        trx_free_for_mysql(trx);
        if (thd != nullptr) {
          thd_to_trx(thd) = nullptr;
        }
      }
    }
  } trx_cleanup{thd, trx, allocated_trx, false};

  struct ThdTxnGuard {
    THD *thd{nullptr};
    bool committed{false};
    ~ThdTxnGuard() {
      if (thd == nullptr || committed) {
        return;
      }
      (void)trans_rollback_stmt(thd);
      (void)trans_rollback(thd);
    }
  } thd_guard{thd, false};

  struct TrxDepthGuard {
    trx_t *trx{nullptr};
    bool entered{false};
    ~TrxDepthGuard() {
      if (entered && trx != nullptr) {
        TrxInInnoDB::end_stmt(trx);
      }
    }
  } depth_guard{trx, false};
  TrxInInnoDB::begin_stmt(trx);
  depth_guard.entered = true;

  handler *aux_h = aux_table->file;
  ha_innobase *aux_innodb = dynamic_cast<ha_innobase *>(aux_h);
  if (aux_innodb == nullptr) {
    ib::warn() << "VECMETA: MEM table handler is not InnoDB for '" << mem_name
               << "'";
    return false;
  }
  if (dynamic_cast<ha_innobase *>(base_table->file) == nullptr) {
    ib::warn() << "VECMETA: base table handler is not InnoDB for '"
               << index->table->name.m_name << "'";
    return false;
  }
  if (aux_table->read_set != nullptr) {
    bitmap_set_all(aux_table->read_set);
  }
  if (aux_table->write_set != nullptr) {
    bitmap_set_all(aux_table->write_set);
  }
  if (base_table->read_set != nullptr) {
    bitmap_set_all(base_table->read_set);
  }

  if (pk_fields >= aux_table->s->fields) {
    ib::warn() << "VECMETA: MEM table layout mismatch for '" << mem_name << "'";
    return false;
  }

  struct RndEndGuard {
    handler *h{nullptr};
    ~RndEndGuard() {
      if (h != nullptr) {
        h->ha_rnd_end();
      }
    }
  };

  Field *seg_field = aux_table->field[pk_fields];
  if (seg_field == nullptr) {
    ib::warn() << "VECMETA: MEM table missing seg_id field for '"
               << mem_name << "'";
    return false;
  }

  const std::string mem_seg_id = seg->vecindex_id;
  if (mem_seg_id.empty()) {
    ib::warn() << "VECMETA: MEM segment missing seg_id for '" << mem_name
               << "'";
    return false;
  }

  if (base_table->s->primary_key == MAX_KEY) {
    ib::warn() << "VECMETA: base table has no primary key; "
               << "point lookup recovery not possible for '"
               << index->table->name.m_name << "'";
    return false;
  }
  if (aux_table->s->primary_key == MAX_KEY) {
    ib::warn() << "VECMETA: MEM table has no primary key for '" << mem_name
               << "'";
    return false;
  }

  KEY *base_pk_info = base_table->key_info + base_table->s->primary_key;
  KEY *aux_pk_info = aux_table->key_info + aux_table->s->primary_key;
  if (pk_fields != static_cast<ulint>(base_pk_info->user_defined_key_parts)) {
    ib::warn() << "VECMETA: base table PK parts mismatch for '"
               << index->table->name.m_name << "' pk_fields=" << pk_fields
               << " pk_parts=" << base_pk_info->user_defined_key_parts;
    return false;
  }
  if (pk_fields != static_cast<ulint>(aux_pk_info->user_defined_key_parts)) {
    ib::warn() << "VECMETA: MEM table PK parts mismatch for '" << mem_name
               << "' pk_fields=" << pk_fields
               << " pk_parts=" << aux_pk_info->user_defined_key_parts;
    return false;
  }
  if (base_pk_info->key_length != aux_pk_info->key_length ||
      base_pk_info->key_length == 0) {
    ib::warn() << "VECMETA: PK key length mismatch for base='"
               << index->table->name.m_name << "' mem='" << mem_name
               << "' base_len=" << base_pk_info->key_length
               << " mem_len=" << aux_pk_info->key_length;
    return false;
  }

  std::vector<uchar> key_buf(static_cast<size_t>(base_pk_info->key_length));

  const ulint vec_field_no =
      dict_table_get_nth_col_pos(clust_index->table, vec_col_no);
  if (vec_field_no == ULINT_UNDEFINED) {
    ib::warn() << "VECMETA: unable to locate vector column in clustered index "
               << "for table '" << index->table->name.m_name << "'";
    return false;
  }

  struct BaseRecTrxIdReader {
    enum class Status { kOk, kNotFound, kInvalidVec, kError };

    dict_index_t *clust_index{nullptr};
    ulint vec_field_no{ULINT_UNDEFINED};
    size_t vec_dim{0};
    trx_t *trx{nullptr};
    mem_heap_t *heap{nullptr};
    dtuple_t *ref{nullptr};
    std::vector<byte> conv_buf;
    ulint n_fields{0};

    BaseRecTrxIdReader(dict_index_t *index, size_t key_len, ulint vec_field,
                       size_t dim, trx_t *trx_in)
        : clust_index(index),
          vec_field_no(vec_field),
          vec_dim(dim),
          trx(trx_in) {
      if (clust_index == nullptr || key_len == 0 ||
          vec_field_no == ULINT_UNDEFINED || vec_dim == 0) {
        return;
      }
      n_fields = dict_index_get_n_unique(clust_index);
      if (n_fields == 0) {
        return;
      }
      heap = mem_heap_create(256, UT_LOCATION_HERE);
      if (heap == nullptr) {
        return;
      }
      ref = dtuple_create(heap, n_fields);
      dict_index_copy_types(ref, clust_index, n_fields);
      conv_buf.resize(key_len);
    }

    ~BaseRecTrxIdReader() {
      if (heap != nullptr) {
        mem_heap_free(heap);
        heap = nullptr;
      }
    }

    Status read(const uchar *key_ptr, size_t key_len, trx_id_t *out_trx_id,
                std::vector<float> &out_vec) {
      if (clust_index == nullptr || ref == nullptr || key_ptr == nullptr ||
          key_len == 0 || conv_buf.size() < key_len ||
          vec_field_no == ULINT_UNDEFINED || vec_dim == 0) {
        return Status::kError;
      }

      row_sel_convert_mysql_key_to_innobase(
          ref, conv_buf.data(), static_cast<ulint>(conv_buf.size()),
          clust_index, key_ptr, static_cast<ulint>(key_len));

      mtr_t mtr;
      mtr_start(&mtr);

      btr_pcur_t pcur;
      const bool found = row_search_on_row_ref(
          &pcur, BTR_SEARCH_LEAF, clust_index->table, ref, &mtr);
      if (!found) {
        pcur.close();
        mtr_commit(&mtr);
        return Status::kNotFound;
      }

      const rec_t *rec = pcur.get_rec();
      if (rec == nullptr || !page_rec_is_user_rec(rec) ||
          rec_get_deleted_flag(rec, dict_table_is_comp(clust_index->table))) {
        pcur.close();
        mtr_commit(&mtr);
        return Status::kNotFound;
      }

      Rec_offsets offsets_holder;
      const ulint *offsets = offsets_holder.compute(rec, clust_index);

      if (out_trx_id != nullptr) {
        *out_trx_id = row_get_rec_trx_id(rec, clust_index, offsets);
      }

      const byte *data = nullptr;
      ulint len = 0;
      mem_heap_t *ext_heap = nullptr;

      if (rec_offs_nth_extern(clust_index, offsets, vec_field_no)) {
        ext_heap = mem_heap_create(1, UT_LOCATION_HERE);
        data = lob::btr_rec_copy_externally_stored_field(
            trx, clust_index, rec, offsets,
            dict_table_page_size(clust_index->table), vec_field_no, &len,
            nullptr, dict_index_is_sdi(clust_index), ext_heap);
      } else {
        data = rec_get_nth_field_instant(rec, offsets, vec_field_no,
                                         clust_index, &len);
      }

      bool vec_ok = true;
      const size_t expected = vec_dim * sizeof(float);
      if (data == nullptr || len == UNIV_SQL_NULL || len != expected) {
        vec_ok = false;
      } else {
        out_vec.resize(vec_dim);
        for (size_t i = 0; i < vec_dim; ++i) {
          float value;
          std::memcpy(&value, data + i * sizeof(float), sizeof(float));
          if (!std::isfinite(value)) {
            vec_ok = false;
            break;
          }
          out_vec[i] = value;
        }
      }

      if (ext_heap != nullptr) {
        mem_heap_free(ext_heap);
      }

      pcur.close();
      mtr_commit(&mtr);

      return vec_ok ? Status::kOk : Status::kInvalidVec;
    }
  };

  BaseRecTrxIdReader base_trx_reader{
      clust_index, static_cast<size_t>(base_pk_info->key_length), vec_field_no,
      dim, trx};

  std::unordered_set<std::string> seen_keys;
  std::vector<float> xb;
  std::vector<int64_t> ids;
  size_t matched_rows = 0;
  /* Segment id currently targeted by the indexed aux scan below; starts as
  the mutable segment and then iterates over the lost birth segments. */
  std::string scan_target = mem_seg_id;

  auto handle_aux_row = [&]() -> bool {
    if (seg_field->is_null()) {
      return true;
    }
    String tmp;
    String* val = seg_field->val_str(&tmp);
    if (val == nullptr) {
      return true;
    }
    const char* row_ptr = val->ptr();
    const size_t row_len = val->length();
    std::string row_seg;
    if (row_ptr != nullptr && row_len != 0) {
      row_seg.assign(row_ptr, row_len);
    }
    if (row_seg != scan_target) {
      return true;
    }

    key_copy(key_buf.data(), aux_table->record[0], aux_pk_info, 0);
    std::string key(reinterpret_cast<const char *>(key_buf.data()),
                    key_buf.size());
    if (!seen_keys.emplace(key).second) {
      return true;
    }
    ++matched_rows;

    trx_id_t creator_trx_id = 0;
    std::vector<float> vec_values;
    const auto read_status = base_trx_reader.read(
        key_buf.data(), key_buf.size(), &creator_trx_id, vec_values);
    if (read_status == BaseRecTrxIdReader::Status::kNotFound) {
      return true;
    }
    if (read_status == BaseRecTrxIdReader::Status::kInvalidVec) {
      ib::warn() << "VECMETA: invalid vector payload during MEM recovery";
      return true;
    }
    if (read_status != BaseRecTrxIdReader::Status::kOk ||
        creator_trx_id == 0) {
      ib::warn() << "VECMETA: failed to read base trx id during MEM recovery "
                 << "for table '" << index->table->name.m_name << "'";
      return false;
    }

    std::vector<vec_pk_column_t> pk_cols;
    bool pk_ok = vec_extract_pk_columns(
        aux_table, clust_index, pk_fields,
        [aux_table](size_t idx) {
          return (idx < static_cast<size_t>(aux_table->s->fields))
                     ? aux_table->field[idx]
                     : nullptr;
        },
        pk_cols);
    if (!pk_ok) {
      ib::warn() << "VECMETA: failed to decode PK from MEM row";
      return false;
    }

    const uint64_t new_id = static_cast<uint64_t>(ids.size());
    ids.push_back(static_cast<int64_t>(new_id));
    xb.insert(xb.end(), vec_values.begin(), vec_values.end());

    dberr_t cache_err = vec_insert_aux_cache(
        &seg->vid_pk_mapping, clust_index, new_id, pk_cols, creator_trx_id);
    if (cache_err != DB_SUCCESS) {
      ib::warn() << "VECMETA: failed to insert pk cache during MEM recovery "
                 << "for id=" << new_id;
      return false;
    }
    return true;
  };

  auto find_seg_index = [&]() -> int {
    if (aux_table->s == nullptr) {
      return -1;
    }
    for (uint i = 0; i < aux_table->s->keys; ++i) {
      KEY *key = aux_table->key_info + i;
      if (key != nullptr && key->name != nullptr &&
          std::strcmp(key->name, "VEC_SEG_ID") == 0) {
        return static_cast<int>(i);
      }
    }
    return -1;
  };

  /* Scan targets: the persisted mutable segment id first, then every birth
  id below it that no Committed or Tombstone manifest entry covers. The
  latter includes ids with a stale non-Committed entry (flush interrupted by
  the crash) and ids with no entry at all (crash between the rotation and
  the first manifest append of the build task). */
  std::vector<std::string> scan_targets;
  scan_targets.push_back(mem_seg_id);
  for (uint32_t cand = 1; cand < seg_id; ++cand) {
    if (covered_seg_ids.count(cand) == 0) {
      scan_targets.push_back(vec_segment_id_from_u32(cand));
    }
  }

  std::unordered_set<uint64_t> adopted_ids(manifest_lost_ids.begin(),
                                           manifest_lost_ids.end());

  bool used_index = false;
  int seg_key = find_seg_index();
  if (seg_key >= 0) {
    KEY *seg_key_info = aux_table->key_info + seg_key;
    if (seg_key_info != nullptr &&
        seg_key_info->user_defined_key_parts > 0) {
      if (aux_h->ha_index_init(seg_key, true) == 0) {
        struct IndexEndGuard {
          handler *h{nullptr};
          ~IndexEndGuard() {
            if (h != nullptr && h->inited != handler::NONE) {
              h->ha_index_end();
            }
          }
        } index_guard{aux_h};

        const CHARSET_INFO* cs = seg_field->charset();
        std::vector<uchar> seg_key_buf(seg_key_info->key_length);

        for (const std::string &target : scan_targets) {
          scan_target = target;
          const size_t rows_before = matched_rows;

          if (seg_field->store(target.c_str(),
                               static_cast<uint>(target.size()),
                               cs != nullptr ? cs : &my_charset_bin) != 0) {
            ib::warn() << "VECMETA: failed to store seg_id key for MEM table '"
                       << mem_name << "'";
            return false;
          }
          key_copy(seg_key_buf.data(), aux_table->record[0], seg_key_info, 0);

          int rc = aux_h->ha_index_read_map(
              aux_table->record[0], seg_key_buf.data(),
              make_prev_keypart_map(seg_key_info->user_defined_key_parts),
              HA_READ_KEY_EXACT);
          if (rc == 0) {
            while (rc == 0) {
              if (!handle_aux_row()) {
                return false;
              }
              rc = aux_h->ha_index_next_same(aux_table->record[0],
                                             seg_key_buf.data(),
                                             seg_key_info->key_length);
            }
            if (rc != HA_ERR_KEY_NOT_FOUND && rc != HA_ERR_END_OF_FILE &&
                rc != HA_ERR_RECORD_IS_THE_SAME) {
              ib::warn()
                  << "VECMETA: index scan failed for MEM table '" << mem_name
                  << "' rc=" << rc;
              return false;
            }
          } else if (rc != HA_ERR_KEY_NOT_FOUND && rc != HA_ERR_END_OF_FILE) {
            ib::warn() << "VECMETA: index read failed for MEM table '"
                       << mem_name << "' rc=" << rc;
            return false;
          }

          if (target != mem_seg_id && matched_rows > rows_before) {
            const uint32_t adopted_num = vec_segment_id_to_u32(target);
            if (adopted_num != 0) {
              adopted_ids.insert(adopted_num);
              ib::warn() << "VECMETA: adopted " << (matched_rows - rows_before)
                         << " aux rows from lost seg_id=" << target
                         << " into mutable segment " << mem_seg_id;
            }
          }
        }
        used_index = true;
      } else {
        ib::warn() << "VECMETA: ha_index_init failed for MEM table '"
                   << mem_name << "', aborting MEM recovery";
      }
    }
  }

  if (!used_index) {
    ib::warn() << "VECMETA: seg_id index unavailable for MEM table '"
               << mem_name << "', aborting MEM recovery";
    return false;
  }

  {
    std::lock_guard<std::shared_mutex> lk(ctx->mu);
    seg->adopted_seg_ids.assign(adopted_ids.begin(), adopted_ids.end());
  }

  if (matched_rows == 0) {
    ib::warn() << "VECMETA: MEM table '" << mem_name
               << "' empty; nothing to recover.";
    return true;
  }

  if (ids.empty()) {
    ib::warn() << "VECMETA: no matching base rows found for MEM recovery on '"
               << index->table->name.m_name << "'";
    return true;
  }

  {
    std::lock_guard<std::shared_mutex> lk(ctx->mu);
    seg->index->add(ids.size(), xb.data(), ids.data());
  }

  seg->vid_pk_mapping.ready = true;
  ctx->id_alloc.next = static_cast<uint64_t>(seg->index->ntotal());
  ctx->id_alloc.inited = true;

  ib::warn() << "VECMETA: recovered " << ids.size()
             << " vectors into mutable MEM index for table '"
             << index->table->name.m_name << "'";
  return true;
}

class VecMetaLoader {
 public:
  static VecMetaLoader &instance() {
    static VecMetaLoader loader;
    return loader;
  }

  ~VecMetaLoader() { stop(); }

  void schedule(dict_index_t *index) {
    if (index == nullptr || index->vec_runtime == nullptr) {
      return;
    }
    vec_index_ctx_t *ctx = index->vec_runtime;
    if (ctx->bootstrap_load_submitted.exchange(true,
                                               std::memory_order_acq_rel)) {
      return;
    }

    ensure_started();
    {
      std::lock_guard<std::mutex> lock(mu);
      const uint64_t key = static_cast<uint64_t>(index->id);
      if (pending.count(key) != 0) {
        return;
      }
      tasks.push(index);
      pending.insert(key);
    }
    cv.notify_one();
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mu);
      if (!running.load(std::memory_order_acquire)) {
        return;
      }
      running.store(false, std::memory_order_release);
    }
    cv.notify_all();
    if (worker.joinable()) {
      worker.join();
    }
  }

 private:
  VecMetaLoader() = default;
  VecMetaLoader(const VecMetaLoader &) = delete;
  VecMetaLoader &operator=(const VecMetaLoader &) = delete;

  void ensure_started() {
    bool expected = false;
    if (!running.compare_exchange_strong(expected, true,
                                         std::memory_order_acq_rel)) {
      return;
    }
    worker = std::thread(&VecMetaLoader::worker_loop, this);
  }

  void worker_loop() {
    bool thread_initialized = !my_thread_init();
    struct ThreadGuard {
      bool ok{false};
      ~ThreadGuard() {
        if (ok) {
          my_thread_end();
        }
      }
    } guard{thread_initialized};

    if (!thread_initialized) {
      running.store(false, std::memory_order_release);
      cv.notify_all();
      return;
    }

    while (true) {
      dict_index_t *index = nullptr;
      {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [this] {
          return !tasks.empty() || !running.load(std::memory_order_acquire);
        });

        if (!running.load(std::memory_order_acquire) && tasks.empty()) {
          return;
        }
        index = tasks.front();
        tasks.pop();
        if (index != nullptr) {
          pending.erase(static_cast<uint64_t>(index->id));
        }
      }

      process(index);
    }
  }

  void process(dict_index_t *index) {
    ib::warn() << "VECMETA: starting bootstrap load for index "
               << (index->name ? index->name : "(null)");

    vec_index_ctx_t *ctx = index != nullptr ? index->vec_runtime : nullptr;
    auto mark_ready = [ctx]() {
      if (ctx != nullptr) {
        ctx->bootstrap_state.store(VecBootstrapState::READY,
                                   std::memory_order_release);
        ctx->bootstrap_loaded.store(true, std::memory_order_release);
      }
    };
    auto mark_failed = [ctx]() {
      if (ctx != nullptr) {
        ctx->bootstrap_state.store(VecBootstrapState::FAILED,
                                   std::memory_order_release);
        ctx->bootstrap_loaded.store(false, std::memory_order_release);
        ctx->bootstrap_load_submitted.store(false, std::memory_order_release);
      }
    };

    if (index == nullptr || ctx == nullptr || index->vec_params == nullptr) {
      mark_failed();
      ib::warn() << "VECMETA: invalid index for bootstrap load.";
      return;
    }

    ib::warn() << "VECMETA: processing bootstrap load for index "
               << (index->name ? index->name : "(null)");
    std::string meta_path;
    VecBgThdGuard guard;
    if (!guard.create()) {
      mark_failed();
      ib::warn() << "VECMETA: failed to create THD for aux cache preload";
      return;
    }

    VecMetaHeader header{};
    std::vector<VecSegmentEntry> entries;
    bool meta_ok = false;
    bool meta_expected = false;
    if (vec_meta_path_for_index(index, &meta_path)) {
      meta_expected = true;
      if (my_access(meta_path.c_str(), F_OK) != 0) {
        meta_expected = false;
      } else if (vec_meta_read_all(meta_path, &header, &entries)) {
        VecMetaHeader expected = vec_meta_make_header(index, ctx->params);
        auto header_matches = [](const VecMetaHeader &lhs,
                                 const VecMetaHeader &rhs) {
          return lhs.magic == rhs.magic && lhs.version == rhs.version &&
                 lhs.index_id == rhs.index_id &&
                 lhs.dimension == rhs.dimension &&
                 lhs.index_type == rhs.index_type &&
                 lhs.metric_type == rhs.metric_type;
        };

        if (!header_matches(header, expected)) {
          meta_expected = true;
          ib::warn() << "VECMETA: header mismatch for index "
                     << (index->name ? index->name : "(null)")
                     << ", skip meta segments.";
        } else {
          meta_ok = true;
        }
      } else {
        ib::warn() << "VECMETA: failed to read meta file '" << meta_path << "'";
      }
    } else {
      ib::warn() << "VECMETA: no meta path available for index "
                 << (index->name ? index->name : "(null)")
                 << "; attempting MEM recovery only.";
    }

    bool ok = true;
    if (meta_expected && !meta_ok) {
      ok = false;
    }

    /* Birth ids whose entries are durable elsewhere (Committed segment file
    or Tombstone after flush/merge carried them forward), and manifest ids
    whose flush never committed — the latter are recovered from the auxiliary
    table together with the mutable segment. */
    std::set<uint32_t> covered_seg_ids;
    std::vector<uint64_t> manifest_lost_ids;

    const std::string base_dir = vec_meta_dirname(meta_path);
    const std::string prefix =
      ctx->index_name_prefix.empty() ? vec_aux_full_name(index)
                                     : ctx->index_name_prefix;

    if (meta_ok) {
      std::unordered_map<uint64_t, VecSegmentEntry> latest_by_id;
      latest_by_id.reserve(entries.size());
      for (const auto &entry : entries) {
        latest_by_id[entry.seg_id] = entry;
      }
      std::vector<VecSegmentEntry> latest_entries;
      latest_entries.reserve(latest_by_id.size());
      for (const auto &kv : latest_by_id) {
        latest_entries.push_back(kv.second);
      }
      std::sort(latest_entries.begin(), latest_entries.end(),
                [](const VecSegmentEntry &a, const VecSegmentEntry &b) {
                  return a.seg_id < b.seg_id;
                });

      ib::warn() << "VECMETA: loading " << latest_entries.size()
                 << " segments (from " << entries.size()
                 << " entries) for index "
                 << (index->name ? index->name : "(null)");

      for (const auto &entry : latest_entries) {
        vec_index_segment_t *inserted = nullptr;
        ib::warn() << "VECMETA: processing segment " << entry.seg_id
                   << " with state " << static_cast<int>(entry.state)
                   << " and file name '" << entry.file_name << "'";
        if (entry.state == static_cast<uint8_t>(VecSegmentState::Tombstone)) {
          covered_seg_ids.insert(static_cast<uint32_t>(entry.seg_id));
          continue;
        }
        if (entry.state != static_cast<uint8_t>(VecSegmentState::Committed)) {
          /* Interrupted flush or merge: any file for this segment may be
          torn and is never read. A merge target contributes no auxiliary
          rows under its own id (its sources stay Committed until the target
          commits), so re-ingesting by birth id is always safe. */
          manifest_lost_ids.push_back(entry.seg_id);
          continue;
        }
        const uint32_t seg_id_num = static_cast<uint32_t>(entry.seg_id);
        covered_seg_ids.insert(seg_id_num);
        const std::string seg_id_str = vec_segment_id_from_u32(seg_id_num);

        {
          std::lock_guard<std::shared_mutex> lk(ctx->mu);
          const bool exists = std::any_of(
              ctx->segments.begin(), ctx->segments.end(),
              [&seg_id_str](const vec_segment_ptr &s) {
                return s && s->vecindex_id == seg_id_str && s->immutable;
              });
          if (exists) {
            continue;
          }
        }

        std::string seg_path = vec_meta_join(base_dir, entry.file_name);
        if (seg_path.empty()) {
          ib::warn() << "VECMETA: empty segment path for seg_id=" << seg_id_num;
          continue;
        }

        std::unique_ptr<IVectorIndex> target;
        switch (ctx->params.backend) {
          case BackendType::Faiss:
            target = vec_make_faiss_index(ctx->params);
            break;
          case BackendType::Hnswlib:
            target = vec_make_hnswlib_index(ctx->params);
            break;
          case BackendType::Diskann:
            target = vec_make_diskann_index(ctx->params);
            break;
          default:
            break;
        }
        if (!target) {
          ib::warn() << "VECMETA: unable to allocate index for seg_id="
                     << seg_id_num;
          continue;
        }
        
        ib::warn() << "VECMETA: loading segment " << seg_id_num;
        const auto load_t0 = std::chrono::steady_clock::now();

        auto loaded = std::make_shared<vec_index_segment_t>();
        loaded->index_file_name = seg_path;
        loaded->vecindex_id = seg_id_str;
        loaded->immutable = true;

        /* The PK mapping and the index file deserialize into disjoint
        objects, so overlap them. */
        const bool expect_pk_map = vec_meta_segment_has_pk_mapping(entry);
        const std::string map_path = vec_vid_pk_mapping_path(seg_path);
        std::future<bool> pkmap_future;
        if (!map_path.empty()) {
          vid_pk_mapping_t *map_out = &loaded->vid_pk_mapping;
          pkmap_future = std::async(std::launch::async, [map_path, map_out]() {
            return vec_vid_pk_mapping_load(map_path, map_out);
          });
        }

        bool index_ok = true;
        try {
          target->load(seg_path);
        } catch (const std::exception &e) {
          index_ok = false;
          ib::warn() << "VECMETA: failed to load segment file '" << seg_path
                     << "' error=" << e.what();
        }

        bool map_ok = false;
        if (pkmap_future.valid()) {
          map_ok = pkmap_future.get();
        }
        if (!index_ok) {
          ok = false;
          continue;
        }
        loaded->index = std::move(target);
        ib::warn() << "VECMETA: segment " << seg_id_num
                   << " index+pkmap loaded in "
                   << std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - load_t0)
                          .count()
                   << " ms";

        if (!map_path.empty()) {
          if (map_ok) {
            ib::warn() << "VECMETA: restored PK mapping from '" << map_path
                       << "' rows=" << loaded->vid_pk_mapping.size();
            loaded->vid_pk_mapping.ready = true;
          } else if (expect_pk_map) {
            ib::warn() << "VECMETA: pk mapping flagged but not readable at '"
                       << map_path << "'";
          }
        }

        std::lock_guard<std::shared_mutex> lk(ctx->mu);
        const bool dup = std::any_of(
            ctx->segments.begin(), ctx->segments.end(),
            [&seg_id_str](const vec_segment_ptr &s) {
              return s && s->vecindex_id == seg_id_str && s->immutable;
            });
        if (!dup) {
          ctx->segments.push_back(std::move(loaded));
          ctx->max_vecindex_id =
              std::max<uint32_t>(ctx->max_vecindex_id, seg_id_num);
          inserted = ctx->segments.back().get();
          vec_commit_topology(ctx);
        }

        ib::warn() << "VECMETA: loaded segment " << seg_id_num
                   << " for index "
                   << (index->name ? index->name : "(null)");

        if (inserted != nullptr && guard.thd() != nullptr &&
            !inserted->vid_pk_mapping.ready) {
              //TODO: May need a flag to resave PK mapping to file?
          if (!vec_load_aux_cache_for_segment(index, ctx, inserted, guard.thd())) {
            ok = false;
          }
        }
      }
    }

    if (guard.thd() != nullptr &&
        !vec_recover_mutable_mem_index(index, ctx, guard.thd(),
                                       covered_seg_ids, manifest_lost_ids)) {
      ib::warn() << "VECMETA: MEM recovery failed for index "
                 << (index->name ? index->name : "(null)");
      ok = false;
    }
    if (guard.thd() == nullptr) {
      ok = false;
    }

    if (ok) {
      mark_ready();
    } else {
      mark_failed();
    }
  }

  std::queue<dict_index_t *> tasks;
  std::set<uint64_t> pending;
  std::mutex mu;
  std::condition_variable cv;
  std::thread worker;
  std::atomic<bool> running{false};
};

void vec_schedule_bootstrap_load(dict_index_t *index) {
  if (index == nullptr || index->vec_runtime == nullptr) {
    return;
  }
  vec_index_ctx_t *ctx = index->vec_runtime;
  VecBootstrapState state =
      ctx->bootstrap_state.load(std::memory_order_acquire);
  if (state == VecBootstrapState::READY ||
      state == VecBootstrapState::LOADING) {
    return;
  }

  if (state == VecBootstrapState::FAILED) {
    ctx->bootstrap_load_submitted.store(false, std::memory_order_release);
  }

  VecBootstrapState expected = state;
  if (!ctx->bootstrap_state.compare_exchange_strong(
          expected, VecBootstrapState::LOADING,
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    return;
  }
  ctx->bootstrap_loaded.store(false, std::memory_order_release);
  VecMetaLoader::instance().schedule(index);
}

void vec_wait_table_builds_idle(dict_table_t *table) {
  if (table == nullptr) {
    return;
  }
  for (dict_index_t *index = table->first_index(); index != nullptr;
       index = index->next()) {
    vec_index_ctx_t *ctx = index->vec_runtime;
    if (ctx == nullptr) {
      continue;
    }
    while (ctx->is_rotation_pending.load(std::memory_order_acquire) ||
           ctx->build_in_progress.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}
