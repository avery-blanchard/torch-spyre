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

from typing import Callable

from sympy import Symbol

from .op_spec import IndexLoad, OpSpec, TensorArg


def has_index_load(arg: TensorArg) -> bool:
    """Return True if any of arg's device_coordinates contains an IndexLoad node."""
    return any(
        coord.has(IndexLoad)
        for coord in arg.device_coordinates
        if hasattr(coord, "has")
    )


def get_index_load_names(arg: TensorArg) -> set[str]:
    """Return the set of tensor names referenced by IndexLoad nodes in arg's coordinates."""
    names: set[str] = set()
    for coord in arg.device_coordinates:
        if not hasattr(coord, "atoms"):
            continue
        for node in coord.atoms(IndexLoad):
            names.add(node.tensor_name)
    return names


def is_indirect_value_tensor(arg: TensorArg) -> bool:
    """Return True if this tensor is accessed via indirect indexing (has IndexLoad in coords)."""
    return has_index_load(arg)


def is_index_tensor(arg: TensorArg, op_spec: OpSpec) -> bool:
    """Return True if this tensor's name is referenced by an IndexLoad in any other arg."""
    name = getattr(arg, "name", "")
    if not name:
        return False
    for other in op_spec.args:
        if other is arg:
            continue
        if name in get_index_load_names(other):
            return True
    return False


def get_index_tensor_for_value(
    op_spec: OpSpec, value_arg: TensorArg
) -> TensorArg | None:
    """Find the index TensorArg referenced by an IndexLoad in value_arg's coordinates."""
    names = get_index_load_names(value_arg)
    for arg in op_spec.args:
        if getattr(arg, "name", "") in names:
            return arg
    return None


def get_value_tensor_idx_for_index(op_spec: OpSpec, index_arg_idx: int) -> int:
    """Return the position of the value tensor accessed by the given index tensor.

    Returns -1 if no corresponding value tensor is found.
    """
    index_arg = op_spec.args[index_arg_idx]
    index_name = getattr(index_arg, "name", "")
    if not index_name:
        return -1
    for j, arg in enumerate(op_spec.args):
        if index_name in get_index_load_names(arg):
            return j
    return -1


def collect_index_tensor_layouts(
    op_spec: OpSpec,
    symbol_mapping: dict,
    index_tensor_indices: set[int],
    logger: object,
) -> tuple[dict, dict]:
    """First pass: compute (dim_order, stick_dim) for each index tensor.

    Returns:
        index_tensor_layouts: dict mapping tensor_idx -> (dim_order, stick_dim)
        index_active_dims: dict mapping tensor_idx -> set of active (non-stick) dims
    """
    from torch_spyre._inductor.codegen.superdsc import _get_device_dim_order

    index_tensor_layouts: dict[int, tuple[list, object]] = {}
    index_active_dims: dict[int, set] = {}

    for i in index_tensor_indices:
        arg = op_spec.args[i]
        dim_order, stick_dim = _get_device_dim_order(arg, symbol_mapping)
        index_tensor_layouts[i] = (dim_order, stick_dim)
        active_dims = {d for d in dim_order if d is not stick_dim}
        index_active_dims[i] = active_dims
        logger.debug(
            f"Index tensor {i}: dim_order={dim_order}, stick_dim={stick_dim}, "
            f"active_dims={sorted(map(str, active_dims))}"
        )

    return index_tensor_layouts, index_active_dims


def _get_index_tensor_device_size_at(
    index_arg: TensorArg,
    stride_idx: int,
) -> int | None:
    """Return index_arg.device_size at the same stride position, or None if out of range."""
    pos = -stride_idx - 2
    if abs(pos) <= len(index_arg.device_size):
        return index_arg.device_size[pos]
    return None


def compute_indirect_max_dim_sizes(
    tensor_idx: int,
    dim: Symbol,
    stick_dim: Symbol | None,
    stride_idx: int,
    original_dev_dim_size: int,
    op_spec: OpSpec,
    symbol_mapping: dict,
    index_tensor_indices: set[int],
    index_active_dims: dict,
    logger: object,
) -> int:
    """Compute max_dim_size for a tensor dimension in an indirect access operation.

    Value tensor:
      - stick dim → -1 (always dynamic)
      - dimension not in index tensor → -1 (data dimension, full range)
      - dimension in index tensor, same size as value → -1 (dynamic)
      - dimension in index tensor, index smaller → 1 (indirect dimension)

    Index tensor:
      - -1 for active (non-stick) dims, 0 for stick dim

    Output tensor:
      - always -1
    """
    arg = op_spec.args[tensor_idx]

    if is_indirect_value_tensor(arg):
        if dim == stick_dim:
            return -1
        index_arg = get_index_tensor_for_value(op_spec, arg)
        if index_arg is None:
            return -1
        idx_size = _get_index_tensor_device_size_at(index_arg, stride_idx)
        if idx_size is None:
            return -1
        if idx_size < original_dev_dim_size:
            return 1
        return -1

    elif tensor_idx in index_tensor_indices:
        active = index_active_dims.get(tensor_idx, set())
        if dim in active:
            return -1
        return 0

    # Output tensor
    return -1


def should_use_kernel_idx_layout(
    tensor_idx: int, index_tensor_indices: set[int]
) -> bool:
    """Return True if this tensor should use KERNEL_IDX layout."""
    return tensor_idx in index_tensor_indices


def get_indirect_access_layout_label(
    is_value_tensor: bool,
    is_output_tensor: bool,
    has_indirect_access: bool,
) -> str | None:
    """Return special layout label for indirect access value/output tensors, or None."""
    if not has_indirect_access:
        return None
    if is_value_tensor:
        return "INPUT"
    if is_output_tensor:
        return "OUTPUT"
    return None


def get_indirect_layout_label(
    tensor_idx: int,
    arg: TensorArg,
    index_tensor_indices: set[int],
    has_indirect_access: bool,
    layouts: dict,
    dim_order: list,
    effective_stick: object,
    stick_size: int,
    layout_labels: list[str],
    get_layout_label_func: Callable,
    logger: object,
) -> str:
    """Get layout label for a tensor in an indirect access operation.

    Index tensors → KERNEL_IDX
    Value tensors → INPUT
    Output tensor → OUTPUT
    Others → fall back to normal layout assignment.
    """

    def ensure_layout(label: str) -> None:
        if label not in layouts:
            layouts[label] = {
                "dim_order": dim_order,
                "stick_dim_order": effective_stick,
                "stick_size": stick_size,
            }

    if should_use_kernel_idx_layout(tensor_idx, index_tensor_indices):
        label = "KERNEL_IDX"
        logger.debug(f"Tensor {tensor_idx}: KERNEL_IDX layout (index tensor)")
        ensure_layout(label)
        return label

    is_value = is_indirect_value_tensor(arg)
    is_output = not arg.is_input
    indirect_label = get_indirect_access_layout_label(
        is_value, is_output, has_indirect_access
    )

    if indirect_label:
        logger.debug(f"Tensor {tensor_idx}: {indirect_label} layout (indirect access)")
        ensure_layout(indirect_label)
        return indirect_label

    return get_layout_label_func(
        layouts, dim_order, effective_stick, stick_size, layout_labels
    )
