# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# FLAGS_pir_apply_shape_optimization_pass=0 FLAGS_enable_pir_api=1
# FLAGS_prim_enable_dynamic=true FLAGS_prim_all=true
# FLAGS_cinn_new_group_scheduler=1 FLAGS_group_schedule_tiling_first=1 FLAGS_cinn_bucket_compile=True
# FLAGS_cinn_compile_with_nvrtc=True FLAGS_nvrtc_compile_to_cubin=True
# FLAGS_support_reduce_stride_read=1

import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, cos, sin, position_ids):
        cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]

        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


def create_tensor_inputs():
    q = paddle.randn([1, 1024, 8, 96], dtype="float32")
    k = paddle.randn([1, 1024, 8, 96], dtype="float32")
    cos = paddle.randn([1, 1024, 1, 96], dtype="float32")
    sin = paddle.randn([1, 1024, 1, 96], dtype="float32")
    position_ids = paddle.randint(high=1024, shape=[1, 1024], dtype="int64")

    inputs = (q, k, cos, sin, position_ids)
    return inputs


# def create_numpy_inputs():
#     q = np.random.normal(size=(13, 1024, 32, 128)).astype("float32")
#     k = np.random.normal(size=(13, 1024, 32, 128)).astype("float32")
#     cos = np.random.normal(size=(1, 1024, 1, 128)).astype("float32")
#     sin = np.random.normal(size=(1, 1024, 1, 128)).astype("float32")
#     position_ids = np.random.normal(0, 1024, size=(13, 1024)).astype("int64")
#     inputs = (q, k, cos, sin, position_ids)
#     return inputs


class PaddleRopeSubGraph(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, cos, sin, position_ids):
        (
            out_q,
            out_k,
            _,
        ) = paddle.incubate.nn.functional.fused_rotary_position_embedding(
            q, k, None, sin, cos, position_ids, use_neox_rotary_style=False
        )
        return out_q, out_k


class TestRopeSubGraph(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        (
            self.q,
            self.k,
            self.cos,
            self.sin,
            self.position_ids,
        ) = create_tensor_inputs()

    def apply_to_static(self, net, use_cinn, input_spec=None):
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def eval(self, use_cinn):
        if use_cinn:
            net = LayerCase()
        else:
            net = PaddleRopeSubGraph()
        net.eval()
        net = self.apply_to_static(net, use_cinn)
        # for i in range(100000):
        #     out = net(self.q, self.k, self.cos, self.sin, self.position_ids)
        import time

        total_time = 0.0
        times = []
        for i in range(5000):
            if i > 100 or i < 9900:
                t0 = time.time()
                out = net(self.q, self.k, self.cos, self.sin, self.position_ids)
                total_time += time.time() - t0
                times.append(time.time() - t0)
        sorted_times = sorted(times)

        def calculate_average(arr):
            return sum(arr) / len(arr) * 1e6

        print(
            'time: ',
            str(total_time / 9000.0 * 1e6),
            " average sorted is: ",
            str(calculate_average(sorted_times[500:-500])),
            " (micro sec)",
        )

        return out

    def test_eval(self):
        cinn_outs = self.eval(use_cinn=True)
        dy_outs = self.eval(use_cinn=False)

        for cinn_out, dy_out in zip(cinn_outs, dy_outs):
            np.testing.assert_allclose(
                cinn_out.numpy(), dy_out.numpy(), atol=1e-6
            )


if __name__ == '__main__':
    unittest.main()
