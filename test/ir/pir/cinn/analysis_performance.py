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

# read file
import matplotlib.pyplot as plt

#
with open(
    '/Paddle/Paddle/paddle/cinn/ir/group_schedule/config/tile_config/NVGPU_NVIDIA_A100_SXM4_40GB/S_R/Sdynamic_Rstatic_vs_static.csv',
    'r',
) as fp:
    data = fp.readlines()


# 创建一个包含2行4列共8个子图的图表
line_num = len(data)
num = int(np.sqrt(line_num))
assert num * num == line_num
fig, axs = plt.subplots(num, num, figsize=(8, 11))


def draw_graph(ax, y, x_label, is_left, is_bottom, is_top):
    # 创建折线图
    x = ['default', 'auto tiling']
    bar_colors = ['tab:blue', 'tab:orange']
    types = ['Defualt config', 'Auto Tiling']
    ax.bar(x, y, color=bar_colors, label=types)
    # 设置图表标题和轴标签
    # ax.set_xlabel(x_label, fontsize=3)
    # ax.set_ylabel('Time(us)', fontsize=3)

    # 设置图例
    # ax.legend()

    # 设置坐标轴范围
    # ax.set_xlim(x_begin, x_end)
    ax.set_ylim(np.min(y) * 0.8, np.max(y) * 1.5)

    if is_bottom == 1:
        # ax.yticks(fontsize=3.9)
        ax.tick_params(axis='both', labelsize=6)
        ax.set_xticks([])
        # ax.set_xlabel(x_label, fontsize=6)
    else:
        ax.tick_params(bottom=False, top=True)
        ax.set_xticks([])
    if is_left == 1:
        ax.set_ylabel('Time(us)', fontsize=6)
        # ax.yticks(fontsize=3.9)
        ax.tick_params(axis='both', labelsize=2)
    else:
        ax.set_yticks([])
        ax.tick_params(axis='both', labelsize=2)
        ax.tick_params(left=True, right=False)
    fontdictdict = {'fontsize': 5}
    if is_top == 1:
        if is_left == 1:
            ax.set_title(
                "R" + x_label.split('R')[1].strip(),
                loc='center',
                y=0.95,
                fontdict=fontdictdict,
            )
        else:
            ax.set_title(
                x_label.split('R')[1].strip().strip(':'),
                loc='center',
                y=0.95,
                fontdict=fontdictdict,
            )
    if is_left == 1:
        if is_top == 1:
            ax.set_ylabel(
                ylabel='Time(us)\n' + x_label.split('R')[0].strip(), fontsize=5
            )
        else:
            ax.set_ylabel(
                ylabel=x_label.split('R')[0].split('S')[1].strip(':'),
                fontsize=5,
            )

    # 显示网格线
    ax.grid(True)


speedup_list = []
for i, piece in enumerate(data):
    label, default, auto, speedup = piece.split(',')
    default = float(default)
    auto = float(auto)
    speedup = float(speedup)
    if auto > default:
        auto = default + 1e-3 * auto  # for handling unstable data
    is_top = 1 if (int(i / num) == 0) else 0
    is_left = 1 if (int(i % num) == 0) else 0
    is_bottom = 1 if (int(i / num) == num - 1) else 0
    is_right = 1 if (int(i % num) == num - 1) else 0
    draw_graph(
        axs[int(i / num), int(i % num)],
        np.array([default, auto]),
        label,
        is_left,
        is_bottom,
        is_top,
    )
    speedup_list.append(speedup)

print(
    f'min max and mean speed up are {np.min(np.array(speedup_list))} (x), {np.max(np.array(speedup_list))} (x), {np.mean(np.array(speedup_list))} (x)'
)
# 调整子图布局
lines = []
labels = []
for ax in axs[0]:
    axLine, axLabel = ax.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)
    break


fig.legend(
    lines,
    labels,
    ncol=2,
    bbox_to_anchor=(0.5, 0.94),
    loc='upper center',
    fontsize=7,
)

print('jok', lines)

# 保存图表为图像文件（例如PNG)

fig.savefig('Sdynamic_Rstatic_performance.png')
fig.savefig(
    'Sdynamic_Rstatic_performance.pdf', format='pdf', bbox_inches='tight'
)

# 显示图表
# plt.show()
