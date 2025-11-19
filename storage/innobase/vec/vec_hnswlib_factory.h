#pragma once
#include <memory>
#include "vec_params.h"
#include "vec_index.h"
#include <external/hnswlib/hnswlib/hnswlib.h>

std::unique_ptr<IVectorIndex> vec_make_hnswlib_index(const vec_params_t& p);
