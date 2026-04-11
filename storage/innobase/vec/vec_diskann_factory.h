#pragma once

#include <memory>

#include "vec_index.h"
#include "vec_params.h"

std::unique_ptr<IVectorIndex> vec_make_diskann_index(const vec_params_t &p);
