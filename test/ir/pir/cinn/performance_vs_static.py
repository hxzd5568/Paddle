# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License');
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
import sys
from os.path import dirname

import numpy as np

# import paddle

sys.path.append(dirname(dirname(__file__)))

# /Paddle/Paddle/paddle/cinn/ir/group_schedule/config/tile_config/NVGPU_NVIDIA_A100_SXM4_40GB/S_R/Sdynamic_Rdynamic_vs_search_st_loc_0.csv
for i in range(1):
    file_path = (
        "/Paddle/Paddle/paddle/cinn/ir/group_schedule/config/tile_config/NVGPU_NVIDIA_A100_SXM4_40GB/S_R/Sdynamic_Rdynamic_vs_search_st_loc_"
        + str(i)
        + ".csv"
    )
    with open(file_path, 'r') as fp:
        data = fp.readlines()
    line_num = len(data)
    num = int(np.sqrt(line_num))
    # assert num*num == line_num
    speedup_list = []
    for i, piece in enumerate(data):
        label, default, auto, speedup = piece.split(',')
        default = float(default)
        auto = float(auto)
        speedup = float(speedup)
        if speedup < 8:
            speedup_list.append(speedup)

    print(
        "location_",
        i,
        f'min max and mean speed up are {np.min(np.array(speedup_list))} (x), {np.max(np.array(speedup_list))} (x), {np.mean(np.array(speedup_list))} (x)',
    )
