add_library(innobase_vec STATIC
  vec_params.cc
  vec_index_runtime.cc
  vec_faiss_factory.cc
  vec_hnswlib_factory.cc
  vec_index_adapter.cc
  vec_meta.cc
  vec_builder.cc
  vec_aux_tables.cc
  vec_id_alloc.cc
  vec_txn_buf.cc
  vec_ingest.cc
  vec_sql.cc
  vec_tasks.cc
)

# only contain innobase and faiss headers
target_include_directories(innobase_vec PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..            # storage/innobase/*
  ${CMAKE_CURRENT_SOURCE_DIR}/../include    # storage/innobase/include/*
  ${CMAKE_SOURCE_DIR}/external/faiss        # submodule headers
  ${CMAKE_SOURCE_DIR}/external/hnswlib      # submodule headers (HNSWLIB)
)

find_package(OpenMP)

# choose the FAISS target: first avx512 -> avx2 -> generic
if (TARGET faiss_avx512)
  set(_FAISS_TGT faiss_avx512)
elseif (TARGET faiss_avx2)
  set(_FAISS_TGT faiss_avx2)
elseif (TARGET faiss)
  set(_FAISS_TGT faiss)
else()
  message(FATAL_ERROR "No FAISS target found. Did you enable -DWITH_FAISS=ON?")
endif()

if (WITH_HNSWLIB)
  if (TARGET hnswlib)
    set(_HNSWLIB_TGT hnswlib)
  else()
    message(FATAL_ERROR "HNSWLIB is enabled but the target 'hnswlib' was not built.")
  endif()
endif()


set(_VECINDEX_LIBRARIES)

if (DEFINED _FAISS_TGT)
  list(APPEND _VECINDEX_LIBRARIES ${_FAISS_TGT})
  message(STATUS "innobase_vec will link with FAISS: ${_FAISS_TGT}")
endif()

if (DEFINED _HNSWLIB_TGT)
  list(APPEND _VECINDEX_LIBRARIES ${_HNSWLIB_TGT})
  message(STATUS "innobase_vec will link with HNSWLIB: ${_HNSWLIB_TGT}")
endif()

if (OpenMP_CXX_FOUND)
  list(APPEND _VECINDEX_LIBRARIES OpenMP::OpenMP_CXX)
  message(STATUS "innobase_vec will link with OpenMP")
else()
  message(WARNING "OpenMP not found; vector backends will run single-threaded")
endif()

if (NOT _VECINDEX_LIBRARIES)
  message(FATAL_ERROR "Neither FAISS nor HNSWLIB targets were found for linking.")
endif()

# 链接所有收集到的库
target_link_libraries(innobase_vec PUBLIC ${_VECINDEX_LIBRARIES})

target_compile_definitions(innobase_vec PRIVATE WITH_VECINDEX=1)


# (Optional) MSVC reduce some warnings
if (MSVC)
  target_compile_options(innobase_vec PRIVATE /W3 /wd4244 /wd4267)
endif()
