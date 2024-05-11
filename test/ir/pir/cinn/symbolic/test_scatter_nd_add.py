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
import sys
import unittest
from os.path import dirname

import numpy as np

import paddle
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))
import utils


def scatter_nd_add(x, index, updates):
    return paddle.scatter_nd_add(x, index, updates)


class CINNSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = scatter_nd_add

    def forward(self, x, y, z):
        out = self.fn(x, y, z)
        return out


class TestCinnSubGraphScatterNdAdd(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        x = np.array([[65, 17], [-14, -25]])
        index = np.array([[], []]).astype('int64')
        updates = np.array([[[-1, -2], [1, 2]], [[3, 4], [-3, -4]]])
        self.index = paddle.to_tensor(index)
        self.x = paddle.to_tensor(x)
        self.updates = paddle.to_tensor(updates)
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = CINNSubGraphNet()
        input_spec = [
            InputSpec(shape=[None, 2], dtype='float32'),
            InputSpec(shape=[2, 0], dtype='int64'),
            InputSpec(shape=[2, None, 2], dtype='float32'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x, self.index, self.updates)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval_symbolic(self):
        dy_out = self.eval_symbolic(use_cinn=False)
        cinn_out = self.eval_symbolic(use_cinn=True)
        print(cinn_out.numpy())
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
