// vec_faiss_factory.h
#pragma once
#include <memory>
#include "vec_params.h"
namespace faiss { struct Index; }
std::unique_ptr<faiss::Index> vec_make_faiss_index(const vec_params_t& p);
