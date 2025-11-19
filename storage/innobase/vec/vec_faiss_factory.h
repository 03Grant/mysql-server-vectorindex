// vec_faiss_factory.h
#pragma once
#include <memory>
#include "vec_params.h"
#include "vec_index.h"

std::unique_ptr<IVectorIndex> vec_make_faiss_index(const vec_params_t& p);
