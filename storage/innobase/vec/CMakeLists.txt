add_library(innobase_vec STATIC
  vec_params.cc
  vec_index_runtime.cc
  vec_faiss_factory.cc
  vec_index_adapter.cc
  vec_builder.cc
  vec_aux_tables.cc
  vec_id_alloc.cc
)

# only contain innobase and faiss headers
target_include_directories(innobase_vec PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..           # storage/innobase/*
  ${CMAKE_CURRENT_SOURCE_DIR}/../include    # storage/innobase/include/*
  ${CMAKE_SOURCE_DIR}/external/faiss       # submodule headers
)

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
message(STATUS "innobase_vec will link with ${_FAISS_TGT}")
target_link_libraries(innobase_vec PUBLIC ${_FAISS_TGT})

target_compile_definitions(innobase_vec PRIVATE WITH_VECINDEX=1)

# (Optional) MSVC reduce some warnings
if (MSVC)
  target_compile_options(innobase_vec PRIVATE /W3 /wd4244 /wd4267)
endif()
