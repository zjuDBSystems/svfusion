# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import glob

template = """/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * NOTE: this file is generated by compute_distance_00_generate.py
 *
 * Make changes there and run in this directory:
 *
 * > python compute_distance_00_generate.py
 *
 */

{includes}

namespace ffanns::neighbors::cagra::detail {{

using namespace ffanns::distance;
{content}

}}  // namespace ffanns::neighbors::cagra::detail
"""

mxdim_team = [(128, 8), (256, 16), (512, 32)]
#mxdim_team = [(64, 8), (128, 16), (256, 32)]
#mxdim_team = [(32, 8), (64, 16), (128, 32)]


# rblock = [(256, 4), (512, 2), (1024, 1)]
# rcandidates = [32]
# rsize = [256, 512]
code_book_types = ["half"]

# search_types = dict(
#     float_uint32=("float", "uint32_t", "float"),  # data_t, idx_t, distance_t
#     half_uint32=("half", "uint32_t", "float"),
#     int8_uint32=("int8_t", "uint32_t", "float"),
#     uint8_uint32=("uint8_t", "uint32_t", "float"),
# )

search_types = dict(
    float_uint32=("float", "uint32_t", "float"),  # data_t, idx_t, distance_t
    uint8_uint32=("uint8_t", "uint32_t", "float")
)

metric_prefix = 'DistanceType::'

specs = []
descs = []
cmake_list = []


# Cleanup first
for f in glob.glob("compute_distance_standard_*.cu"):
  os.remove(f)
for f in glob.glob("compute_distance_vpq_*.cu"):
  os.remove(f)

# Generate new files
for type_path, (data_t, idx_t, distance_t) in search_types.items():
    for (mxdim, team) in mxdim_team:
        # CAGRA
        for metric in ['L2Expanded', 'InnerProduct']:
            path = f"compute_distance_standard_{metric}_{type_path}_dim{mxdim}_t{team}.cu"
            includes = '#include "compute_distance_standard-impl.cuh"'
            params = f"{metric_prefix}{metric}, {team}, {mxdim}, {data_t}, {idx_t}, {distance_t}"
            spec = f"standard_descriptor_spec<{params}>"
            content = f"""template struct {spec};"""
            specs.append(spec)
            with open(path, "w") as f:
                f.write(template.format(includes=includes, content=content))
                cmake_list.append(f"  src/neighbors/detail/cagra/{path}")

with open("compute_distance-ext.cuh", "w") as f:
    includes = '''
#pragma once

#include "compute_distance_standard.hpp"
'''
    newline = "\n"
    contents = f'''
{newline.join(map(lambda s: "extern template struct " + s + ";", specs))}

extern template struct
  instance_selector<{("," + newline + "                    ").join(specs)}>;

using descriptor_instances =
  instance_selector<{("," + newline + "                    ").join(specs)}>;

template <typename DataT, typename IndexT, typename DistanceT, typename DatasetT>
auto dataset_descriptor_init(const cagra::search_params& params,
                             const DatasetT& dataset,
                             ffanns::distance::DistanceType metric)
  -> dataset_descriptor_host<DataT, IndexT, DistanceT>
{{
  auto [init, priority] = descriptor_instances::select<DataT, IndexT, DistanceT>(params, dataset, metric);
  if (init == nullptr || priority < 0) {{
    RAFT_FAIL("No dataset descriptor instance compiled for this parameter combination.");
  }}
  return init(params, dataset, metric);
}}
'''
    f.write(template.format(includes=includes, content=contents))


with open("compute_distance.cu", "w") as f:
    includes = '#include "compute_distance-ext.cuh"'
    newline = "\n"
    contents = f'''
template struct instance_selector<{("," + newline + "                    ").join(specs)}>;
'''
    f.write(template.format(includes=includes, content=contents))

cmake_list.sort()
for path in cmake_list:
    print(path)
