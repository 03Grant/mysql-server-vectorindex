#pragma once

#include <memory>
#include <string>
#include <vector>

#include "vec_index.h"
#include "vec_params.h"

std::unique_ptr<IVectorIndex> vec_make_diskann_index(const vec_params_t &p);
bool vec_diskann_has_native_artifacts(const std::string &prefix);
void vec_diskann_collect_artifact_paths(const std::string &prefix,
                                        std::vector<std::string> *out);
void vec_diskann_remove_artifacts(const std::string &prefix);
bool vec_diskann_build_from_fbin(const vec_params_t &params, size_t points_num,
                                 const std::string &data_path,
                                 const std::string &prefix,
                                 std::string *error_out);
