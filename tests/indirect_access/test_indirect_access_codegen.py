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

"""
Unit tests for indirect access code generation.

Covers:
1. IndexLoad detection in device_coordinates
2. Index/value tensor identification
3. maxDimSizes computation
4. Layout label assignment (INPUT, KERNEL_IDX, OUTPUT)
5. Address assignment via SEGMENT_OFFSETS
6. Data format handling

NOTE: Run with pytest, not directly with python:
    pytest tests/indirect_access/test_indirect_access_codegen.py -v
"""

import pytest
from sympy import Symbol, floor, Mod, Integer

from torch_spyre._C import DataFormats
from torch_spyre._inductor.op_spec import IndexLoad, OpSpec, TensorArg
from torch_spyre._inductor.indirect_access import (
    collect_index_tensor_layouts,
    compute_indirect_max_dim_sizes,
    get_indirect_tensor_address,
    has_index_load,
    get_index_load_names,
    is_index_tensor,
    is_indirect_value_tensor,
)


class MockLogger:
    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


mock_logger = MockLogger()


def _make_identity_op_spec():
    """Build the two-OpSpec indirect access example from PR #2656.

    Arg 0: index tensor (name='arg1_1', IEEE_INT32)
    Arg 1: value tensor (SEN169_FP16, has IndexLoad('arg1_1') in coordinates)
    Arg 2: output tensor (SEN169_FP16)
    """
    c0 = Symbol("c0")
    c1 = Symbol("c1")
    c2 = Symbol("c2")

    index_arg = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=DataFormats.IEEE_INT32,
        device_size=[1, 6, 3, 32],
        device_coordinates=[
            Integer(0),
            floor(c1 / 32),
            c0,
            Mod(c1, 32),
        ],
        allocation={"hbm": 17179869184},
        stride_map=[32, 32, 192, 32],
        name="arg1_1",
    )

    value_arg = TensorArg(
        is_input=True,
        arg_index=1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[1, 4, 128, 64],
        device_coordinates=[
            Integer(0),
            floor(c2 / 64),
            IndexLoad("arg1_1"),
            Mod(c2, 64),
        ],
        allocation={"hbm": 34359738368},
        stride_map=[64, 64, 256, 64],
    )

    output_arg = TensorArg(
        is_input=False,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[192, 4, 3, 64],
        device_coordinates=[c1, floor(c2 / 64), c0, Mod(c2, 64)],
        allocation={"pool": 0},
        stride_map=[256, 64, 49152, 64],
    )

    op_spec = OpSpec(
        op="identity",
        is_reduction=False,
        iteration_space={
            c0: (Integer(3), 1),
            c1: (Integer(192), 1),
            c2: (Integer(256), 1),
        },
        op_info={},
        args=[index_arg, value_arg, output_arg],
    )
    return op_spec, index_arg, value_arg, output_arg


class TestIndexLoadDetection:
    """Test detection of IndexLoad nodes in device_coordinates."""

    def test_has_index_load_on_value_tensor(self):
        _, _, value_arg, _ = _make_identity_op_spec()
        assert has_index_load(value_arg) is True

    def test_has_index_load_false_on_index_tensor(self):
        _, index_arg, _, _ = _make_identity_op_spec()
        assert has_index_load(index_arg) is False

    def test_has_index_load_false_on_output_tensor(self):
        _, _, _, output_arg = _make_identity_op_spec()
        assert has_index_load(output_arg) is False

    def test_get_index_load_names(self):
        _, _, value_arg, _ = _make_identity_op_spec()
        names = get_index_load_names(value_arg)
        assert names == {"arg1_1"}

    def test_get_index_load_names_empty_for_plain_tensor(self):
        _, index_arg, _, _ = _make_identity_op_spec()
        assert get_index_load_names(index_arg) == set()

    def test_is_indirect_value_tensor(self):
        _, _, value_arg, _ = _make_identity_op_spec()
        assert is_indirect_value_tensor(value_arg) is True

    def test_is_indirect_value_tensor_false(self):
        _, index_arg, _, output_arg = _make_identity_op_spec()
        assert is_indirect_value_tensor(index_arg) is False
        assert is_indirect_value_tensor(output_arg) is False

    def test_is_index_tensor(self):
        op_spec, index_arg, _, _ = _make_identity_op_spec()
        assert is_index_tensor(index_arg, op_spec) is True

    def test_is_index_tensor_false_for_value(self):
        op_spec, _, value_arg, output_arg = _make_identity_op_spec()
        assert is_index_tensor(value_arg, op_spec) is False
        assert is_index_tensor(output_arg, op_spec) is False

    def test_is_index_tensor_false_when_no_name(self):
        """Tensors without a name are never identified as index tensors."""
        op_spec, _, value_arg, _ = _make_identity_op_spec()
        # value_arg has no name field, so should not be an index tensor
        assert is_index_tensor(value_arg, op_spec) is False


class TestAddressAssignment:
    """Test SEGMENT_OFFSETS address assignment for indirect access tensors."""

    def test_index_tensor_gets_segment_1(self):
        from torch_spyre._inductor.constants import SEGMENT_OFFSETS

        op_spec, index_arg, _, _ = _make_identity_op_spec()
        addr = get_indirect_tensor_address(index_arg, 0, op_spec)
        assert addr == SEGMENT_OFFSETS[1]

    def test_value_tensor_gets_segment_0(self):
        from torch_spyre._inductor.constants import SEGMENT_OFFSETS

        op_spec, _, value_arg, _ = _make_identity_op_spec()
        addr = get_indirect_tensor_address(value_arg, 1, op_spec)
        assert addr == SEGMENT_OFFSETS[0]

    def test_output_tensor_gets_segment_2(self):
        from torch_spyre._inductor.constants import SEGMENT_OFFSETS

        op_spec, _, _, output_arg = _make_identity_op_spec()
        addr = get_indirect_tensor_address(output_arg, 2, op_spec)
        assert addr == SEGMENT_OFFSETS[2]


class TestMaxDimSizesComputation:
    """Test maxDimSizes computation for indirect access tensors."""

    def test_value_tensor_all_dims_dynamic(self):
        """Value tensor dims should all be -1 (dynamic)."""
        op_spec, index_arg, value_arg, _ = _make_identity_op_spec()
        c0 = Symbol("c0")
        c1 = Symbol("c1")
        c2 = Symbol("c2")
        symbol_mapping = {c0: c0, c1: c1, c2: c2}
        index_tensor_indices = {0}

        _, index_active_dims = collect_index_tensor_layouts(
            op_spec, symbol_mapping, index_tensor_indices, mock_logger
        )

        # value tensor is arg index 1; its dims are c1 (via floor) and c2 (via Mod/stick)
        # The IndexLoad dimension is not in dim_order, so only c2-related dims are tested
        for dim, dev_size in [(c1, 4), (c2, 64)]:
            result = compute_indirect_max_dim_sizes(
                tensor_idx=1,
                dim=dim,
                stick_dim=c2,
                original_dev_dim_size=dev_size,
                op_spec=op_spec,
                symbol_mapping=symbol_mapping,
                index_tensor_indices=index_tensor_indices,
                index_active_dims=index_active_dims,
                logger=mock_logger,
            )
            assert result == -1, f"dim {dim}: expected -1, got {result}"

    def test_index_tensor_active_dims_dynamic(self):
        """Active dims of an index tensor should be -1; stick dim should be 0."""
        op_spec, _, _, _ = _make_identity_op_spec()
        c0 = Symbol("c0")
        c1 = Symbol("c1")
        c2 = Symbol("c2")
        symbol_mapping = {c0: c0, c1: c1, c2: c2}
        index_tensor_indices = {0}

        _, index_active_dims = collect_index_tensor_layouts(
            op_spec, symbol_mapping, index_tensor_indices, mock_logger
        )

        # index tensor (arg 0): active dims are c0 and c1 (the non-stick dims)
        # stick dim (innermost of the index tensor) gets 0
        for dim in (c0, c1):
            result = compute_indirect_max_dim_sizes(
                tensor_idx=0,
                dim=dim,
                stick_dim=c1,
                original_dev_dim_size=3,
                op_spec=op_spec,
                symbol_mapping=symbol_mapping,
                index_tensor_indices=index_tensor_indices,
                index_active_dims=index_active_dims,
                logger=mock_logger,
            )
            assert result == -1, f"index dim {dim}: expected -1, got {result}"

    def test_output_tensor_dims_always_dynamic(self):
        """Output tensor max_dim_sizes should always be -1."""
        op_spec, _, _, _ = _make_identity_op_spec()
        c0 = Symbol("c0")
        c1 = Symbol("c1")
        c2 = Symbol("c2")
        symbol_mapping = {c0: c0, c1: c1, c2: c2}
        index_tensor_indices = {0}

        _, index_active_dims = collect_index_tensor_layouts(
            op_spec, symbol_mapping, index_tensor_indices, mock_logger
        )

        result = compute_indirect_max_dim_sizes(
            tensor_idx=2,
            dim=c1,
            stick_dim=c2,
            original_dev_dim_size=4,
            op_spec=op_spec,
            symbol_mapping=symbol_mapping,
            index_tensor_indices=index_tensor_indices,
            index_active_dims=index_active_dims,
            logger=mock_logger,
        )
        assert result == -1


class TestLayoutLabelAssignment:
    """Test layout label assignment for indirect access tensors."""

    def test_index_tensor_is_detected_for_kernel_idx(self):
        """Index tensors should be identified by is_index_tensor()."""
        op_spec, index_arg, value_arg, output_arg = _make_identity_op_spec()
        assert is_index_tensor(index_arg, op_spec) is True
        assert is_index_tensor(value_arg, op_spec) is False
        assert is_index_tensor(output_arg, op_spec) is False

    def test_value_tensor_detected_for_input_label(self):
        """Value tensors (with IndexLoad) should be detected as indirect value tensors."""
        _, _, value_arg, _ = _make_identity_op_spec()
        assert is_indirect_value_tensor(value_arg) is True

    def test_output_tensor_not_value_tensor(self):
        _, _, _, output_arg = _make_identity_op_spec()
        assert is_indirect_value_tensor(output_arg) is False


class TestDataFormatHandling:
    """Test data format assignment for indirect access tensors."""

    def test_index_tensor_dtype_is_int32(self):
        """Index tensors in the new encoding use IEEE_INT32 device dtype."""
        _, index_arg, _, _ = _make_identity_op_spec()
        # The SDSC layer re-labels this to SENUINT32, but the TensorArg itself is IEEE_INT32
        assert index_arg.device_dtype == DataFormats.IEEE_INT32

    def test_value_tensor_preserves_fp16(self):
        _, _, value_arg, _ = _make_identity_op_spec()
        assert value_arg.device_dtype == DataFormats.SEN169_FP16


class TestCollectIndexTensorLayouts:
    """Test the first-pass layout collection for index tensors."""

    def test_collect_returns_dim_order_and_active_dims(self):
        op_spec, _, _, _ = _make_identity_op_spec()
        c0 = Symbol("c0")
        c1 = Symbol("c1")
        c2 = Symbol("c2")
        symbol_mapping = {c0: c0, c1: c1, c2: c2}
        index_tensor_indices = {0}

        layouts, active_dims = collect_index_tensor_layouts(
            op_spec, symbol_mapping, index_tensor_indices, mock_logger
        )

        assert 0 in layouts
        dim_order, stick_dim = layouts[0]
        assert isinstance(dim_order, list)
        assert len(dim_order) > 0

        assert 0 in active_dims
        assert isinstance(active_dims[0], set)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
