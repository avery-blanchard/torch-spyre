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

"""Reorder non-stick dimensions to satisfy operation requirements.

The three-pass restickify pipeline (propagate_layouts -> optimize_restickify ->
insert_restickify) resolves stick-dimension layout constraints. Some operations
impose additional requirements on non-stick dimension ordering.

This pass runs after insert_restickify, once every op has a committed
FixedTiledLayout. For operations with nonstick-dim-order requirements, it checks
whether the value tensor's current dim_order matches; if not, either rewrites the
producer's output layout in place (single-consumer case) or inserts a copy in
 the required layout. New requirement sources can be added by extending
_get_nonstick_dim_order_requirements().
"""

import sympy

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, MutationLayoutSHOULDREMOVE

from torch_spyre._C import SpyreTensorLayout, get_device_dtype

from .constants import ELIDED_COPY_BACK_ATTR
from .errors import Unsupported
from .insert_restickify import _fixed_tiled, insert_restickify_on_node_inputs
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .op_spec import IndirectAccess
from .pass_utils import device_coordinates, indirect_info_from_op

logger = get_inductor_logger("reorder_nonstick_dims")


def _real_layout(buf) -> FixedTiledLayout:
    layout = buf.get_layout()
    if isinstance(layout, MutationLayoutSHOULDREMOVE):
        assert getattr(buf, ELIDED_COPY_BACK_ATTR, False), (
            f"unexpected mutation layout on {buf.get_name()!r}"
        )
        layout = layout.real_layout()
    return layout


def _indirect_stride_idx(coords: list[sympy.Expr], access_subs: dict) -> int | None:
    """Return the stride_idx (from right, 0-indexed) of the first IndirectAccess
    coordinate, or None if coords carry no indirect symbol."""
    for idx, coord in enumerate(reversed(coords)):
        substituted = coord.xreplace(access_subs) if access_subs else coord
        if hasattr(substituted, "has") and substituted.has(IndirectAccess):
            return idx
    return None


def _dim_order_is_compliant(value_stl: SpyreTensorLayout, stride_idx: int) -> bool:
    """Check if indirect access is at the outermost (leftmost) device position.

    For indirect access to work correctly, the IndirectAccess coordinate must
    be at device position 0 (the outermost dimension before non-stick and stick).
    This corresponds to stride_idx being positioned such that the indirect
    dimension is leftmost.
    """
    v_n = len(value_stl.stride_map)
    v_indirect_pos = v_n - 1 - stride_idx

    # Compliant if indirect is at position 0 (outermost device dim)
    compliant = v_indirect_pos == 0

    logger.debug(
        "_dim_order_is_compliant: v_n=%d, stride_idx=%d, indirect_pos=%d, compliant=%s",
        v_n,
        stride_idx,
        v_indirect_pos,
        compliant,
    )
    return compliant


def _stride_map_to_host_dim(
    stride_map: list, host_stride: list, device_pos: int
) -> int | None:
    """Match a device dim's stride_map entry to its host dim by stride value.

    Mirrors pass_utils.py::lower_pad_sequence's sm_last heuristic: the
    within-stick (last) device dim's stride_map entry equals the host
    stride of the dim it tiles, and non-stick device dims are never split,
    so their stride_map entries equal a host stride directly too. Returns
    None if no host dim has a matching stride (e.g. a synthetic/padded
    device dim with stride_map entry <= 0).
    """
    sm = int(stride_map[device_pos])
    if sm <= 0:
        return None
    return next((d for d, s in enumerate(host_stride) if int(s) == sm), None)


def _build_required_stl(
    value_layout,
    required_dim_order: list[int],
) -> SpyreTensorLayout:
    """Build a new STL with the indirect dim outermost (device position 0).

    For rank >= 3, the host-side constructor's dim_order param does this
    directly. Rank 2 needs a hand-built device_size/stride_map instead: the
    C++ get_generic_stick_layout() always maps a 2D dim_order to
    dim_map=[last, first, last], i.e. the stick's outer tile always comes
    before the other non-stick dim in dim_map -- there is no dim_order that
    puts a non-stick dim ahead of the stick's outer tile at rank 2. So we
    construct the [non_stick, outer_stick, inner_stick] device layout
    directly via the device_size/stride_map constructor overload, mirroring
    what dim_map_to_stride_map would produce for that layout.
    """
    if len(value_layout.size) == 2:
        from torch_spyre._C import get_elem_in_stick

        eps = get_elem_in_stick(value_layout.dtype)
        host_size_0, host_size_1 = (int(s) for s in value_layout.size)
        host_stride_0, host_stride_1 = (int(s) for s in value_layout.stride)

        device_size = [host_size_0, (host_size_1 + eps - 1) // eps, eps]
        stride_map = [host_stride_0, host_stride_1 * eps, host_stride_1]
        return SpyreTensorLayout(
            device_size, stride_map, get_device_dtype(value_layout.dtype)
        )

    return SpyreTensorLayout(
        value_layout.size,
        value_layout.stride,
        value_layout.dtype,
        required_dim_order,
    )


def _required_dim_order(
    value_stl: SpyreTensorLayout,
    value_layout,
    stride_idx: int,
) -> list[int]:
    """Build the dim_order so indirect access is at device position 0 (outermost).

    Maps each device position to its host dim via stride_map, then reorders
    so the indirect dimension's host dim appears first in the dim_order.

    Raises Unsupported if stride_map entries cannot be resolved to host dims.
    """
    n = len(value_stl.stride_map)
    stride_map = list(value_stl.stride_map)
    host_stride = [int(s) for s in value_layout.stride]

    def _host_dim(device_pos: int) -> int:
        host_dim = _stride_map_to_host_dim(stride_map, host_stride, device_pos)
        if host_dim is None:
            raise Unsupported(
                f"indirect layout: device dim {device_pos} (stride_map entry "
                f"{stride_map[device_pos]}) does not match a host stride in "
                f"{host_stride!r}"
            )
        return host_dim

    stick_dim = _host_dim(n - 1)
    indirect_dim = _host_dim(n - 1 - stride_idx)

    dim_order = [indirect_dim]
    dim_order.extend(
        d for d in range(len(value_layout.size)) if d not in (indirect_dim, stick_dim)
    )
    dim_order.append(stick_dim)
    return dim_order


def _consumer_count(graph: GraphLowering, name: str) -> int:
    """Count read+write references to buffer `name` across graph.operations."""
    count = 0
    for op in graph.operations:
        rw = op.get_read_writes()
        for dep in rw.reads:
            if isinstance(dep, MemoryDep) and dep.name == name:
                count += 1
        for dep in rw.writes:
            if isinstance(dep, MemoryDep) and dep.name == name:
                count += 1
    return count


def _can_mutate_producer_in_place(graph: GraphLowering, value_buf) -> bool:
    if not isinstance(value_buf, ComputedBuffer):
        return False
    if isinstance(value_buf.layout, MutationLayoutSHOULDREMOVE):
        return False
    if value_buf.get_name() in graph.get_output_names():
        return False
    return _consumer_count(graph, value_buf.get_name()) <= 1


def _rewrite_producer_layout(value_buf, required_stl: SpyreTensorLayout) -> None:
    value_buf.layout = _fixed_tiled(value_buf.get_layout(), required_stl)
    logger.info(
        "enforce_indirect_layout: rewrote producer %s layout in place -> %s",
        value_buf.get_name(),
        list(required_stl.stride_map),
    )


def _insert_relayout_copy(
    graph: GraphLowering,
    consumer_op: ComputedBuffer,
    value_buf,
    required_layout: FixedTiledLayout,
) -> ComputedBuffer:
    """Insert a spyre.restickify copy of value_buf in required_layout ahead of
    consumer_op, and patch consumer_op's inner_fn to read the new buffer.

    Returns the reconstructed ComputedBuffer that replaced consumer_op in
    graph.operations (insert_restickify_on_node_inputs invalidates the
    original instance).
    """
    operations = graph.operations
    arg_name = value_buf.get_name()
    consumer_name = consumer_op.get_name()
    insert_restickify_on_node_inputs(
        consumer_op,
        [{"arg_name": arg_name, "target_layout": required_layout}],
        operations,
    )
    logger.info(
        "enforce_indirect_layout: inserted relayout copy of %s before %s",
        arg_name,
        consumer_name,
    )
    return next(
        o
        for o in operations
        if isinstance(o, ComputedBuffer) and o.get_name() == consumer_name
    )


def _value_bufs_for_op(
    graph: GraphLowering,
    op: ComputedBuffer,
    access_subs: dict,
    sizes: dict | None,
) -> list:
    """Return the value-tensor buffers this op indirectly reads (gather:
    any read dep whose device_coordinates contain an IndirectAccess)."""
    value_bufs: list = []
    for dep in op.get_read_writes().reads:
        if not isinstance(dep, MemoryDep):
            continue
        buf = graph.get_buffer(dep.name)
        layout = _real_layout(buf)
        if not isinstance(layout, FixedTiledLayout):
            continue
        coords = [
            c.xreplace(access_subs)
            for c in device_coordinates(layout.device_layout, dep, sizes)
        ]
        if any(hasattr(c, "has") and c.has(IndirectAccess) for c in coords):
            value_bufs.append(buf)
    return value_bufs


def _get_nonstick_dim_order_requirements(
    op: ComputedBuffer,
) -> tuple[set[str], dict, dict[sympy.Symbol, int] | None] | None:
    """Extract non-stick dimension ordering requirements from an operation.

    Returns (dep_names, access_subs, sizes) if the op has requirements, else None.
    Currently checks: indirect-access requirements (via indirect_info_from_op).
    Future: extend to check other requirement sources.
    """
    dep_names, access_subs, sizes = indirect_info_from_op(op)
    if dep_names:
        logger.debug(
            "reorder_nonstick_dims: op %s has nonstick-dim requirements from %d deps",
            op.get_name(),
            len(dep_names),
        )
        return dep_names, access_subs, sizes
    return None


def reorder_nonstick_dims(graph: GraphLowering) -> None:
    """Reorder non-stick dimensions to satisfy operation requirements.

    Runs after insert_restickify: every op's layout is a committed
    FixedTiledLayout at this point. For each operation with non-stick dim-order
    requirements (currently: indirect-access ops), checks whether the value
    tensor's current non-stick dim_order matches what's required; if not,
    either rewrites the producer's layout in place (single-consumer,
    non-mutation, non-graph-output case) or inserts a spyre.restickify copy
    node ahead of the consumer.

    Extensible: new requirement sources can be added by extending
    _get_nonstick_dim_order_requirements().
    """
    for original_op in list(graph.operations):
        if not isinstance(original_op, ComputedBuffer):
            continue
        requirement = _get_nonstick_dim_order_requirements(original_op)
        if not requirement:
            continue
        dep_names, access_subs, sizes = requirement

        op = original_op
        value_bufs = _value_bufs_for_op(graph, op, access_subs, sizes)
        for value_buf in value_bufs:
            value_layout = _real_layout(value_buf)
            if not isinstance(value_layout, FixedTiledLayout):
                continue
            value_stl = value_layout.device_layout

            value_dep = next(
                d
                for d in op.get_read_writes().reads
                if isinstance(d, MemoryDep) and d.name == value_buf.get_name()
            )
            value_coords = device_coordinates(value_stl, value_dep, sizes)
            stride_idx = _indirect_stride_idx(value_coords, access_subs)
            if stride_idx is None:
                continue

            index_names = {
                sym.args[0].name
                for sym in access_subs.values()
                if isinstance(sym, IndirectAccess)
            }
            index_name = next(iter(index_names & dep_names), None)
            if index_name is None:
                continue
            index_dep = next(
                (
                    d
                    for d in op.get_read_writes().reads
                    if isinstance(d, MemoryDep) and d.name == index_name
                ),
                None,
            )
            if index_dep is None:
                continue
            index_layout = _real_layout(graph.get_buffer(index_name))
            if not isinstance(index_layout, FixedTiledLayout):
                continue
            if _dim_order_is_compliant(value_stl, stride_idx):
                continue

            required_dim_order = _required_dim_order(
                value_stl,
                value_layout,
                stride_idx,
            )
            required_stl = _build_required_stl(value_layout, required_dim_order)

            if _can_mutate_producer_in_place(graph, value_buf):
                _rewrite_producer_layout(value_buf, required_stl)
            else:
                required_layout = _fixed_tiled(value_layout, required_stl)
                op = _insert_relayout_copy(graph, op, value_buf, required_layout)
