#include "vec_aux_tables.h"
#include "univ.i"
#include "dict0dict.h"      // dict_table_t, dict_index_t, table_id_t
#include "dict0dd.h"         // dd_table_open_on_name_in_mem, dd_create_vec_index_table ...
#include "dict0mem.h"       // dict_mem_table_create/add_col, dict_mem_index_create ...
#include "row0mysql.h"      // row_create_table_for_mysql, row_create_index_for_mysql
#include "mem0mem.h"        // mem_heap_create/free
#include "ut0ut.h"          // ib::info, ib::warn


#include <cstring>
#include <string>
#include <sstream>

static std::string vec_extract_db(const char* table_name) {
  if (!table_name) return {};
  const char* slash = std::strchr(table_name, '/');
  if (!slash) return {};
  return std::string(table_name, slash - table_name);
}

static std::string vec_full_name(const std::string& db, const std::string& aux) {
  std::string full;
  full.reserve(db.size() + 1 + aux.size());
  full.append(db).push_back('/');
  full.append(aux);
  return full;
}

static bool is_gen_clust_name(const dict_index_t* idx) {
  if (!idx || !idx->name) return false;
  return std::strcmp(idx->name, "GEN_CLUST_INDEX") == 0;
}


/** Extract only the required flags from table->flags2 for FTS Aux
tables.
@param[in]      flags2  Table flags2
@return extracted flags2 for FTS aux tables */
static inline uint32_t vec_get_table_flags2_for_aux_tables(uint32_t flags2) {
  /* Extract the file_per_table flag, temporary file flag and encryption flag
  from the main FTS table flags2 */
  return ((flags2 & DICT_TF2_USE_FILE_PER_TABLE) |
          (flags2 & DICT_TF2_ENCRYPTION_FILE_PER_TABLE) |
          (flags2 & DICT_TF2_TEMPORARY) | DICT_TF2_AUX);
}

/** Create dict_table_t object for FTS Aux tables.
@param[in]      aux_table_name  FTS Aux table name
@param[in]      table           table object of FTS Index
@param[in]      n_cols          number of columns for FTS Aux table
@return table object for FTS Aux table */
static dict_table_t *vec_create_in_mem_aux_table(const char *aux_table_name,
                                                 const dict_table_t *table,
                                                 ulint n_cols) {
  dict_table_t *new_table = dict_mem_table_create(
      aux_table_name, table->space, n_cols, 0, 0, table->flags,
      vec_get_table_flags2_for_aux_tables(table->flags2));

  if (DICT_TF_HAS_SHARED_SPACE(table->flags)) {
    ut_ad(table->space == fil_space_get_id_by_name(table->tablespace()));
    new_table->tablespace = mem_heap_strdup(new_table->heap, table->tablespace);
  }

  if (DICT_TF_HAS_DATA_DIR(table->flags)) {
    ut_ad(table->data_dir_path != nullptr);
    new_table->data_dir_path =
        mem_heap_strdup(new_table->heap, table->data_dir_path);
  }

  return (new_table);
}


static std::string vec_aux_full_name(const dict_index_t* index) {
  ut_ad(index && index->table && index->table->name.m_name);
  // 内部全名：db/I_VEC_<table_id>_<index_id>
  const char* parent = index->table->name.m_name; // 形如 "db/table"
  const char* slash  = std::strchr(parent, '/');
  std::string db = (slash ? std::string(parent, size_t(slash - parent)) : "");
  std::string aux;
  aux.reserve(db.size() + 1 + 32);
  aux.append(db).push_back('/');
  aux.append("I_VEC_")
     .append(std::to_string((unsigned long long)index->table->id))
     .append("_")
     .append(std::to_string((unsigned long long)index->id));
  return aux;
}

/** Update DD for the single auxiliary table of a VEC index.
@param[in]  index   the vector index instance (already has aux table created)
@return DB_SUCCESS or error code */
static dberr_t vec_create_one_index_dd_tables(const dict_index_t* index)
{
    ut_ad(index != nullptr);
    ut_ad(index->table != nullptr);
    ib::warn() << "VECINDEX: DD register step 1! ";
    // 如果你有 DICT_VECINDEX 标志，做个断言
    ut_ad(index->type & DICT_VECINDEX);

    const std::string full_name = vec_aux_full_name(index);
    
    ib::warn() << "VECINDEX: DD register step 2! full_name=" << full_name;
    // 打开 InnoDB 内部已创建好的物理表（只在内存里用，不登记/修改）
    dict_table_t* aux = dd_table_open_on_name_in_mem(full_name.c_str(), false);
    if (aux == nullptr) {
        ib::warn() << "VECINDEX: DD register failed; cannot open aux table in mem: "
                << full_name;
        return DB_FAIL;
    }
    ib::warn() << "VECINDEX: DD register step 3! ";
    // 把 aux 的定义写进 SQL-DD（实现应仿 dd_create_fts_index_table）
    bool ok = dd_create_vec_index_table(index->table, aux);
    if (!ok) {
        ib::warn() << "VECINDEX: dd_create_vec_index_table() failed for " << full_name;
        dd_table_close(aux, nullptr, nullptr, false);
        return DB_FAIL;
    }
    ib::warn() << "VECINDEX: DD register step 4! ";
    dd_table_close(aux, nullptr, nullptr, false);
    ib::warn() << "VECINDEX: DD register successed! ";
    return DB_SUCCESS;
}


/** Check if a table has vector index needs to have its auxiliary index
tables' metadata updated in DD
@param[in,out]  table           table to check
@return DB_SUCCESS or error code */
dberr_t vec_create_index_dd_tables(dict_table_t *table) {
  dberr_t error = DB_SUCCESS;

  for (dict_index_t *index = table->first_index();
       index != nullptr && error == DB_SUCCESS; index = index->next()) {
    if ((index->type & DICT_VECINDEX) && index->fill_dd) {
      ib::warn() << "VECINDEX: creating DD for aux table of index "
                 << (index->name ? index->name : "(null)")
                 << " on table " << (table->name.m_name ? table->name.m_name : "(null)");
      error = vec_create_one_index_dd_tables(index);
      index->fill_dd = false;
    }

    ut_ad(!index->fill_dd);
  }

  return (error);
}


/** 按“基表聚簇索引复制列定义”的规则创建一张 VEC 附属表。
    - 显式主键：逐列复制 mtype/prtype/len/列名，作为 PRIMARY KEY 列集
    - 无显式主键（GEN_CLUST_INDEX）：使用 row_id BIGINT UNSIGNED 作为 PRIMARY KEY
    - 额外加一列：faiss_id BIGINT UNSIGNED UNIQUE
    - 表的 flags 继承自基表（行格式/压缩策略一致）
  @return 成功返回新表指针；失败返回 nullptr（并设置 trx->error_state） */
static dict_table_t* vec_create_one_index_table_pk_compatible(
    trx_t*              trx,
    const dict_index_t* vec_index,         // 该向量索引（含指向基表的 table*）
    const char*         base_table_name,   // "db/table"
    table_id_t          /*table_id*/,      // 未直接使用；表名由 aux_name 唯一化
    const std::string&  aux_name)          // "I_VEC_<tid>_<iid>"
{
  ut_ad(trx != nullptr);
  ut_ad(vec_index != nullptr);
  ut_ad(base_table_name != nullptr);

  dict_table_t* base = vec_index->table;
  if (!base) {
    ib::warn() << "VECINDEX: base table is null";
    return nullptr;
  }

  dict_index_t* clust = base->first_index();
  if (!clust || !clust->is_clustered()) {
    ib::warn() << "VECINDEX: no clustered index found on base table";
    return nullptr;
  }

  mem_heap_t* heap = mem_heap_create(2048, UT_LOCATION_HERE);

  // 1) 计算附属表内部全名：db/aux
  const std::string db = vec_extract_db(base_table_name);
  if (db.empty()) {
    ib::warn() << "VECINDEX: cannot extract db from base=" 
               << (base_table_name ? base_table_name : "(null)");
    mem_heap_free(heap);
    return nullptr;
  }
  const std::string full_name = vec_full_name(db, aux_name);

  // 2) 计算主键列数 / 是否使用 row_id
  bool use_row_id = false;
  ulint pk_n_fields = 0;
  if (std::strcmp(clust->name, "GEN_CLUST_INDEX") == 0) {
    use_row_id   = true;
    pk_n_fields  = 1;                  // row_id
  } else {
    use_row_id   = false;
    pk_n_fields  = clust->n_fields;    // 显式主键列数
  }

  // 3) 准备 dict_mem_table_create() 所需参数
  const ulint     n_cols       = pk_n_fields + 1;  // + faiss_id


  // 4) in-mem 创建表对象
  dict_table_t* new_table = vec_create_in_mem_aux_table(full_name.c_str(), base, n_cols);

  if (!new_table) {
    ib::warn() << "VECINDEX: dict_mem_table_create failed for " << full_name;
    mem_heap_free(heap);
    return nullptr;
  }

  // 5) 加列：先按主键列拷贝，再追加 faiss_id
  if (use_row_id) {
    // row_id BIGINT UNSIGNED NOT NULL
    dict_mem_table_add_col(new_table, heap,
                           "row_id",
                           DATA_INT,
                           DATA_NOT_NULL | DATA_UNSIGNED,
                           8 /* BIGINT */,
                           true);
  } else {
    for (ulint i = 0; i < pk_n_fields; ++i) {
      dict_field_t*    f        = clust->get_field(i);
      const dict_col_t* c       = f ? f->col : nullptr;
      const char*       colname = f ? static_cast<const char*>(f->name) : nullptr;

      if (!c || !colname) {
        ib::warn() << "VECINDEX: invalid PK meta at i=" << i;
        mem_heap_free(heap);
        return nullptr;
      }

      dict_mem_table_add_col(new_table, heap,
                             colname,
                             c->mtype,   // DATA_INT / DATA_VARMYSQL / ...
                             c->prtype,  // NOT_NULL / BINARY / charset bits
                             c->len,     // 变长用最大长度
                             true);
    }
  }

  // 追加 faiss_id BIGINT UNSIGNED NOT NULL
  dict_mem_table_add_col(new_table, heap,
                         "faiss_id",
                         DATA_INT,
                         DATA_NOT_NULL | DATA_UNSIGNED,
                         8 /* BIGINT */,
                         true);

  // 6) 物理建表
  dberr_t err = row_create_table_for_mysql(new_table, nullptr, nullptr, trx, nullptr);
  if (err != DB_SUCCESS) {
    ib::warn() << "VECINDEX: row_create_table_for_mysql failed for " << full_name
               << " err=" << (int)err;
    trx->error_state = err;
    mem_heap_free(heap);
    return nullptr;
  }

  // 7) 建 PRIMARY（聚簇索引）
  {
    dict_index_t* pk = dict_mem_index_create(
        full_name.c_str(),
        "PRIMARY",
        new_table->space,
        DICT_UNIQUE | DICT_CLUSTERED,
        pk_n_fields);

    if (use_row_id) {
      pk->add_field("row_id", 0, true);
    } else {
      for (ulint i = 0; i < pk_n_fields; ++i) {
        dict_field_t* f = clust->get_field(i);
        ulint prefix = f->prefix_len;          // 复制原前缀长度
        bool asc     = f->is_ascending;      // 复制升降序（若你版本没有该方法，按字段 flag 取）
        pk->add_field(f->name, prefix, asc);
      }
    }

    trx_dict_op_t saved = trx_get_dict_operation(trx);
    err = row_create_index_for_mysql(pk, trx, nullptr, nullptr);
    trx->dict_operation = saved;

    if (err != DB_SUCCESS) {
      ib::warn() << "VECINDEX: create PRIMARY on " << full_name
                 << " failed, err=" << (int)err;
      trx->error_state = err;
      mem_heap_free(heap);
      return nullptr;
    }
  }

  // 8) 建唯一二级索引：u_faiss_id(faiss_id)
  {
    dict_index_t* uk = dict_mem_index_create(
        full_name.c_str(),
        "u_faiss_id",
        new_table->space,
        DICT_UNIQUE,
        1);
    uk->add_field("faiss_id", 0, true);

    trx_dict_op_t saved = trx_get_dict_operation(trx);
    err = row_create_index_for_mysql(uk, trx, nullptr, nullptr);
    trx->dict_operation = saved;

    if (err != DB_SUCCESS) {
      ib::warn() << "VECINDEX: create unique index u_faiss_id on " << full_name
                 << " failed, err=" << (int)err;
      trx->error_state = err;
      mem_heap_free(heap);
      return nullptr;
    }
  }

  mem_heap_free(heap);
  return new_table;
}

// --- 入口 ---

dberr_t vec_create_index_tables_low(trx_t* trx,
                                    dict_index_t* index,
                                    const char* table_name,
                                    table_id_t table_id)
{
    if (!trx || !index || !table_name) return DB_FAIL;

    // 唯一且可追溯的附属表名：I_VEC_<table_id>_<index_id>
    std::ostringstream oss;
    oss << "I_VEC_" << static_cast<unsigned long long>(table_id)
        << "_"      << static_cast<unsigned long long>(index->id);
    const std::string aux = oss.str();

    dict_table_t* t = vec_create_one_index_table_pk_compatible(
        trx, index, table_name, table_id, aux);

    ib::warn() << "VECINDEX: creating aux table " << aux
                << " for base=" << table_name
                    << " (index_id=" << (unsigned long long)index->id << ")";
    if (!t) return DB_FAIL;

    index->fill_dd = true; // 与 FTS 逻辑一致：请求填充 DD

    ib::warn() << "VECINDEX: created aux table " << aux
                << " for base=" << table_name
                << " (index_id=" << (unsigned long long)index->id << ")";

    return DB_SUCCESS;
}
