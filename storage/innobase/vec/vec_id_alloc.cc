#include "vec_id_alloc.h"

#include <algorithm>
#include <cstring>
#include <string>

#include "btr0pcur.h"
#include "dict0dict.h"
#include "mach0data.h"
#include "mem0mem.h"
#include "mtr0mtr.h"
#include "page0page.h"
#include "rem0rec.h"
#include "ut0ut.h"

#ifndef UNIV_MUST_NOT_LOG
  #include "ut0dbg.h"     // ut_ad
#endif

// 你可换成自己的日志设施
namespace { inline void log_info(const char* msg)  { ib::info() << msg;  }
           inline void log_warn(const char* msg)  { ib::warn() << msg; } }

namespace {

/** Build auxiliary table internal name for a vector index.
@return fully qualified name like db/I_VEC_<tid>_<iid>, or empty if unavailable */
inline std::string vec_aux_full_name(const dict_index_t *index) {
  if (index == nullptr || index->table == nullptr ||
      index->table->name.m_name == nullptr) {
    return {};
  }

  const char *base = index->table->name.m_name;  // "db/table"
  const char *slash = std::strchr(base, '/');
  if (slash == nullptr || slash == base) {
    return {};
  }

  std::string db(base, static_cast<size_t>(slash - base));
  if (db.empty()) {
    return {};
  }

  std::string full;
  full.reserve(db.size() + 1 + 32);
  full.append(db).push_back('/');
  full.append("I_VEC_")
      .append(std::to_string(static_cast<unsigned long long>(index->table->id)))
      .append("_")
      .append(std::to_string(static_cast<unsigned long long>(index->id)));
  return full;
}

/** Fetch the largest value from a secondary index whose first column is an
unsigned integer. */
inline bool vec_index_last_uint64(dict_index_t *sec_index, uint64_t *value_out) {
  if (value_out == nullptr || sec_index == nullptr) {
    return false;
  }

  *value_out = 0;

  btr_pcur_t pcur;
  mtr_t mtr;
  mtr_start(&mtr);

  bool found = false;

  pcur.open_at_side(false, sec_index, BTR_SEARCH_LEAF, true, 0, &mtr);

  if (!page_is_empty(pcur.get_page())) {
    const rec_t *rec = nullptr;

    do {
      rec = pcur.get_rec();
      if (rec != nullptr && page_rec_is_user_rec(rec)) {
        break;
      }
    } while (pcur.move_to_prev(&mtr));

    if (rec != nullptr && page_rec_is_user_rec(rec)) {
      ulint offsets_buf[REC_OFFS_NORMAL_SIZE];
      ulint *offsets = offsets_buf;
      mem_heap_t *heap = nullptr;

      rec_offs_init(offsets_buf);
      offsets = rec_get_offsets(rec, sec_index, offsets, ULINT_UNDEFINED,
                                UT_LOCATION_HERE, &heap);

      ulint len = 0;
      const byte *data = reinterpret_cast<const byte *>(
          rec_get_nth_field(nullptr, rec, offsets, 0, &len));

      if (data != nullptr && len != UNIV_SQL_NULL && len > 0 && len <= 8) {
        uint64_t value = 0;
        for (ulint i = 0; i < len; ++i) {
          value = (value << 8) | static_cast<uint64_t>(data[i]);
        }
        *value_out = value;
        found = true;
      } else {
        ib::warn() << "VECINDEX: unexpected field size when reading aux index"
                   << " len=" << len;
      }

      if (heap != nullptr) {
        mem_heap_free(heap);
      }
    }
  }

  pcur.close();
  mtr_commit(&mtr);

  return found;
}

}  // namespace

// ==================== 对外方法 ====================

dberr_t vec_id_allocator_t::recover_from_aux(trx_t* trx, dict_index_t* index) {
  if (inited) return DB_SUCCESS;

  std::lock_guard<std::mutex> lk(mu);
  if (inited) return DB_SUCCESS;

  uint64_t max_id = 0;
  bool empty = false;
  dberr_t  e = vec_aux_select_max_id(trx, index, &max_id, &empty);
  if (e != DB_SUCCESS) {
    log_warn("VECINDEX: recover_from_aux failed to read MAX(faiss_id)");
    return e;
  }

  // next = max；若表空（max=0）也安全（初始从 0 开始）
  if (max_id == std::numeric_limits<uint64_t>::max()) {
    log_warn("VECINDEX: aux MAX(faiss_id) at UINT64_MAX; cannot continue");
    return DB_ERROR;
  }
  if(!empty) {
    next   = max_id + 1;
  } else {
    next   = 0;
  }
  inited = true;
  log_info("VECINDEX: id allocator initialized (lazy) from aux table");
  return DB_SUCCESS;
}

dberr_t vec_id_allocator_t::reserve(uint64_t k, uint64_t* start) {
  if (start == nullptr || k == 0) return DB_ERROR;

  std::lock_guard<std::mutex> lk(mu);
  if (!inited) {
    // 你也可以在这里直接返回错误，或触发一次 recover_from_aux(nullptr, index)
    // 为了稳妥，这里返回错误，让上层确保先 recover。
    log_warn("VECINDEX: reserve() called before recover_from_aux()");
    return DB_ERROR;
  }

  // 溢出防护：next + k - 1 <= UINT64_MAX
  if (k > std::numeric_limits<uint64_t>::max() - next) {
    log_warn("VECINDEX: reserve() overflow");
    return DB_ERROR;
  }

  *start = next;
  next  += k;
  return DB_SUCCESS;
}

// ==================== 需要你打通的钩子（占位） ====================

dberr_t vec_aux_select_max_id(trx_t* /*trx*/, dict_index_t* index, uint64_t* out_max, bool* empty) {
  if (out_max == nullptr || index == nullptr || index->table == nullptr) {
    return DB_ERROR;
  }

  *out_max = 0;

  const std::string aux_name = vec_aux_full_name(index);
  if (aux_name.empty()) {
    ib::warn() << "VECINDEX: failed to derive aux table name for index '"
               << (index->name ? index->name : "(null)") << "'";
    return DB_ERROR;
  }

  dict_table_t *aux_table = dict_table_open_on_name(
      aux_name.c_str(), false, false, DICT_ERR_IGNORE_NONE);
  if (aux_table == nullptr) {
    ib::warn() << "VECINDEX: cannot open aux table '" << aux_name << "'";
    return DB_ERROR;
  }

  dict_index_t *faiss_uidx =
      dict_table_get_index_on_name(aux_table, "u_faiss_id");
  if (faiss_uidx == nullptr) {
    ib::warn() << "VECINDEX: aux table '" << aux_name
               << "' missing unique index u_faiss_id";
    dict_table_close(aux_table, false, false);
    return DB_ERROR;
  }

  uint64_t max_id = 0;
  const bool found = vec_index_last_uint64(faiss_uidx, &max_id);

  dict_table_close(aux_table, false, false);

  if (!found) {
    *out_max = 0;
    *empty = true;
    return DB_SUCCESS;
  }

  *out_max = max_id;
  return DB_SUCCESS;
}
