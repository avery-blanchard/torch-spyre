# Copyright 2025 The Torch-Spyre Authors.
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

from types import MethodType, SimpleNamespace
from unittest.mock import PropertyMock, patch

import sympy

from torch_spyre._C import DataFormats
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.spyre_kernel import RValue, SpyreKernel


class _FakeDeviceLayout:
    def __init__(self, device_size, stride_map, device_dtype):
        self.device_size = device_size
        self.stride_map = stride_map
        self.device_dtype = device_dtype


class _FakeLayout(FixedTiledLayout):
    def __init__(self, size, stride, device_size, stride_map, allocation=None):
        self.size = size
        self.stride = stride
        self.allocation = allocation or {"hbm": 0}
        self.per_tile_fixed = False
        self.device_layout = _FakeDeviceLayout(
            device_size=device_size,
            stride_map=stride_map,
            device_dtype=DataFormats.SEN169_FP16,
        )


class _FakeBuffer:
    def __init__(self, layout):
        self._layout = layout

    def get_layout(self):
        return self._layout


class _DummyValue(RValue):
    pass


def test_store_indirect_access_builds_index_tensor_args_inline():
    x0 = sympy.Symbol("x0")
    x1 = sympy.Symbol("x1")

    value_layout = _FakeLayout(
        size=[4, 64],
        stride=[64, 1],
        device_size=[4, 64],
        stride_map=[64, 1],
    )
    index_layout = _FakeLayout(
        size=[4],
        stride=[1],
        device_size=[4],
        stride_map=[1],
    )
    output_layout = _FakeLayout(
        size=[4, 64],
        stride=[64, 1],
        device_size=[4, 64],
        stride_map=[64, 1],
    )

    buffers = {
        "value_tensor": _FakeBuffer(value_layout),
        "index_tensor": _FakeBuffer(index_layout),
        "out": _FakeBuffer(output_layout),
    }

    kernel = SpyreKernel()
    kernel.args = SimpleNamespace(output=lambda name: None)
    kernel.op_specs = []
    kernel.spyre_kernel_args = []
    kernel.current_node = SimpleNamespace(
        node=SimpleNamespace(
            data=SimpleNamespace(
                op_info={
                    "op": "identity",
                    "tensor_names": ["value_tensor", "index_tensor"],
                    "index_args": [1],
                    "index_value_pairs": [{"index_arg": 1, "value_arg": 0}],
                }
            )
        )
    )

    created_args = []

    def fake_create_tensor_arg(
        is_input,
        name,
        tensor,
        is_index_tensor=False,
        related_value_tensor_idx=-1,
    ):
        arg = TensorArg(
            is_input=is_input,
            arg_index=len(created_args),
            device_dtype=tensor.layout.device_layout.device_dtype,
            device_size=list(tensor.layout.device_layout.device_size),
            device_coordinates=[tensor.index],
            allocation=tensor.layout.allocation,
            stride_map=list(tensor.layout.device_layout.stride_map),
            is_index_tensor=is_index_tensor,
            related_value_tensor_idx=related_value_tensor_idx,
        )
        created_args.append((name, arg, tensor.index))
        return arg

    captured = {}

    def fake_create_op_spec(self, op, is_reduction, args, op_info):
        captured["op"] = op
        captured["is_reduction"] = is_reduction
        captured["args"] = args
        captured["op_info"] = dict(op_info)
        return OpSpec(
            op=op,
            is_reduction=is_reduction,
            iteration_space={},
            args=args,
            op_info=op_info,
        )

    fake_graph = SimpleNamespace(
        get_buffer=lambda name: buffers[name],
        sizevars=SimpleNamespace(precomputed_replacements={}),
        scheduler=SimpleNamespace(mutation_real_name={}),
        removed_buffers=set(),
    )

    with (
        patch(
            "torch_spyre._inductor.spyre_kernel.V.__class__.graph",
            new_callable=PropertyMock,
            return_value=fake_graph,
        ),
        patch(
            "torch_spyre._inductor.spyre_kernel.iteration_space",
            return_value={x0: (4, 1), x1: (64, 1)},
        ),
        patch(
            "torch_spyre._inductor.spyre_kernel.enrich_indirect_index_value_pairs"
        ) as enrich_mock,
    ):
        kernel.create_tensor_arg = fake_create_tensor_arg
        kernel.create_op_spec = MethodType(fake_create_op_spec, kernel)
        kernel.store("out", sympy.Integer(0), _DummyValue())

    enrich_mock.assert_called_once_with(
        kernel.current_node.node.data.op_info,
        ["x0", "x1"],
    )

    assert len(kernel.op_specs) == 1
    assert captured["op"] == "identity"
    assert captured["is_reduction"] is False
    assert len(captured["args"]) == 3

    value_arg = captured["args"][0]
    index_arg = captured["args"][1]
    output_arg = captured["args"][2]

    assert value_arg.is_input is True
    assert value_arg.is_index_tensor is False
    assert value_arg.related_value_tensor_idx == -1

    assert index_arg.is_input is True
    assert index_arg.is_index_tensor is True
    assert index_arg.related_value_tensor_idx == 0

    assert output_arg.is_input is False
    assert output_arg.is_index_tensor is False
    assert output_arg.related_value_tensor_idx == -1

    assert created_args[0][0] == "value_tensor"
    assert created_args[1][0] == "index_tensor"
    assert created_args[2][0] == "out"

    assert created_args[0][2] == x0 * 64 + x1
    assert created_args[1][2] == x0
    assert created_args[2][2] == sympy.Integer(0)
