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
impose additional requirements on non-stick dimension ordering based on their
coordinate access patterns. Currently, indirect-access operations (gather: x[i],
torch.gather, index_select; scatter: scatter_, index_put, index_copy, ...) require
that the value tensor's non-stick free-symbol dims align positionally with the
index tensor's non-stick dims (right-to-left, excluding stick). This requirement
is currently only assumed at SuperDSC codegen time (indirect_access.py, used by
codegen/supydsc.py) and never validated pre-scheduler; a violation silently
produces wrong max_dim_sizes/strides in codegen.

This pass runs after insert_restickify, once every op has a committed
FixedTiledLayout. For operations with nonstick-dim-order requirements, it checks
whether the value tensor's current dim_order matches; if not, either rewrites the
producer's output layout in place (single-consumer case) or inserts a
spyre.restickify copy in the required layout (mirroring insert_restickify.py's
own mechanism). New requirement sources can be added by extending
_get_nonstick_dim_order_requirements().
"""

import sympy

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, MutationLayoutSHOULDREMOVE

from torch_spyre._C import SpyreTensorLayout

from .constants import ELIDED_COPY_BACK_ATTR
from .errors import Unsupported
from .insert_restickify import _fixed_tiled, insert_restickify_on_node_inputs
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .op_spec import IndirectAccess
from .pass_utils import (
    _find_scatter_index_buf_names,
    device_coordinates,
    indirect_info_from_op,
)

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


def _indirect_dim_symbols(
    index_coords: list[sympy.Expr], access_subs: dict
) -> list[sympy.Symbol]:
    """Ordered, deduped free symbols from index_coords, right-to-left, excluding stick.

    Mirrors indirect_access.py::get_indirect_dim_symbols one stage earlier.
    """
    seen: set = set()
    ordered: list = []
    for i in range(len(index_coords) - 2, -1, -1):
        expr = index_coords[i].xreplace(access_subs) if access_subs else index_coords[i]
        for sym in getattr(expr, "free_symbols", ()):
            if sym not in seen:
                seen.add(sym)
                ordered.append(sym)
    return ordered


def _value_dim_symbols(
    value_coords: list[sympy.Expr], access_subs: dict, stick_idx_from_right: int
) -> list[sympy.Symbol]:
    """Ordered, deduped free symbols from value_coords, right-to-left, excluding
    stick and excluding the indirect-access position itself."""
    n = len(value_coords)
    indirect_pos = n - 1 - stick_idx_from_right
    seen: set = set()
    ordered: list = []
    for i in range(n - 2, -1, -1):
        if i == indirect_pos:
            continue
        expr = value_coords[i].xreplace(access_subs) if access_subs else value_coords[i]
        for sym in getattr(expr, "free_symbols", ()):
            if sym not in seen:
                seen.add(sym)
                ordered.append(sym)
    return ordered


def _dim_order_is_compliant(
    value_coords: list[sympy.Expr],
    index_coords: list[sympy.Expr],
    access_subs: dict,
    stride_idx: int,
) -> bool:
    """Check that value's non-stick dim order (right-to-left) matches index's."""
    index_syms = _indirect_dim_symbols(index_coords, access_subs)
    value_syms = _value_dim_symbols(value_coords, access_subs, stride_idx)
    return value_syms[: len(index_syms)] == index_syms


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


def _required_dim_order(
    value_stl: SpyreTensorLayout,
    value_layout,
    value_coords: list[sympy.Expr],
    index_coords: list[sympy.Expr],
    access_subs: dict,
    stride_idx: int,
) -> list[int]:
    """Build the dim_order value_stl must use to satisfy the index tensor's
    non-stick ordering, keeping the currently committed stick dim fixed.

    Resolves each device dim's host dim by matching value_stl.stride_map
    against value_layout's host strides (mirrors pass_utils.py's
    lower_pad_sequence sm_last heuristic), since SpyreTensorLayout does not
    expose dim_order back after construction.

    Raises Unsupported if the index tensor's active-dim count does not fit
    the value tensor's non-stick rank, or if a device dim's stride_map entry
    cannot be resolved to a host dim.
    """
    n = len(value_coords)
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

    index_syms = _indirect_dim_symbols(index_coords, access_subs)
    # Map each host free symbol in value_coords back to its host dim index.
    sym_to_dim: dict = {}
    for device_pos in range(n - 1):
        if device_pos == n - 1 - stride_idx:
            continue
        coord = value_coords[device_pos]
        substituted = coord.xreplace(access_subs) if access_subs else coord
        host_dim = _host_dim(device_pos)
        for sym in getattr(substituted, "free_symbols", ()):
            sym_to_dim.setdefault(sym, host_dim)

    ordered_dims = []
    for sym in index_syms:
        dim = sym_to_dim.get(sym)
        if dim is None:
            raise Unsupported(
                f"indirect layout: index symbol {sym} has no corresponding "
                f"host dim in value tensor coords {value_coords!r}"
            )
        if dim not in ordered_dims:
            ordered_dims.append(dim)

    remaining = [
        d
        for d in range(n)
        if d != stick_dim and d != indirect_dim and d not in ordered_dims
    ]
    dim_order = ordered_dims + remaining + [indirect_dim] + [stick_dim]
    if len(dim_order) != n or set(dim_order) != set(range(n)):
        raise Unsupported(
            f"indirect layout: could not construct a valid dim_order from "
            f"index symbols {index_syms!r} and value rank {n}"
        )
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


# Sentinel returned by _value_bufs_for_op in place of the op's own output
# buffer (scatter self-mutation case). We cannot return `op` itself and
# compare by identity later, because a gather value_buf processed earlier in
# the same outer iteration may have already reconstructed `op` into a fresh
# ComputedBuffer instance (see CLAUDE.md's "wrap, never reconstruct" note in
# insert_restickify_on_node_inputs) -- identity would then silently break.
_OWN_OUTPUT = object()


def _value_bufs_for_op(
    graph: GraphLowering,
    op: ComputedBuffer,
    dep_names: set,
    access_subs: dict,
) -> list:
    """Return the value-tensor buffers this op indirectly accesses.

    Gather: any read dep whose device_coordinates contain an IndirectAccess.
    Scatter: _OWN_OUTPUT, when the op's write dep is indirect and an index
    buffer name was found via _find_scatter_index_buf_names.
    """
    value_bufs: list = []
    rw = op.get_read_writes()
    for dep in rw.reads:
        if not isinstance(dep, MemoryDep):
            continue
        buf = graph.get_buffer(dep.name)
        layout = _real_layout(buf)
        if not isinstance(layout, FixedTiledLayout):
            continue
        coords = device_coordinates(layout.device_layout, dep, None)
        if any(
            hasattr(c.xreplace(access_subs), "has")
            and c.xreplace(access_subs).has(IndirectAccess)
            for c in coords
        ):
            value_bufs.append(buf)

    scatter_index_names = _find_scatter_index_buf_names(op)
    if scatter_index_names:
        write_dep = next(iter(rw.writes))
        if isinstance(write_dep, MemoryDep) and write_dep.is_indirect():
            value_bufs.append(_OWN_OUTPUT)

    return value_bufs


def _get_nonstick_dim_order_requirements(
    op: ComputedBuffer,
) -> tuple[set[str], dict, dict[sympy.Symbol, int] | None] | None:
    """Extract non-stick dimension ordering requirements from an operation.

    Returns (dep_names, access_subs, sizes) if the op has requirements, else None.
    Currently checks: indirect-access requirements (via indirect_info_from_op).
    Future: extend to check other requirement sources.
    """
    # Check indirect-access requirements
    dep_names, access_subs, sizes = indirect_info_from_op(op)
    if dep_names:
        return dep_names, access_subs, sizes
    # Future: check other requirement sources here
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

        # _insert_relayout_copy reconstructs the consumer ComputedBuffer (per
        # CLAUDE.md's "wrap, never reconstruct inner_fn" rule); track the
        # live instance so a second value_buf in this same op sees the
        # already-patched reads/writes rather than a stale snapshot.
        op = original_op
        value_bufs = _value_bufs_for_op(graph, op, dep_names, access_subs)
        for value_buf in value_bufs:
            is_own_output = value_buf is _OWN_OUTPUT
            resolved_value_buf = op if is_own_output else value_buf
            value_layout = _real_layout(resolved_value_buf)
            if not isinstance(value_layout, FixedTiledLayout):
                continue
            value_stl = value_layout.device_layout

            if is_own_output:
                value_dep = next(
                    d for d in op.get_read_writes().writes if isinstance(d, MemoryDep)
                )
            else:
                value_dep = next(
                    d
                    for d in op.get_read_writes().reads
                    if isinstance(d, MemoryDep)
                    and d.name == resolved_value_buf.get_name()
                )
            value_coords = device_coordinates(value_stl, value_dep, sizes)
            stride_idx = _indirect_stride_idx(value_coords, access_subs)
            if stride_idx is None:
                continue

            index_names = {
                sym.args[0].name
                for sym in access_subs.values()
                if isinstance(sym, IndirectAccess)
            } | _find_scatter_index_buf_names(op)
            index_name = next(iter(index_names & dep_names), None)
            if index_name is None:
                continue
            index_buf = graph.get_buffer(index_name)
            index_layout = _real_layout(index_buf)
            if not isinstance(index_layout, FixedTiledLayout):
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
            index_coords = device_coordinates(
                index_layout.device_layout, index_dep, sizes
            )

            if _dim_order_is_compliant(
                value_coords, index_coords, access_subs, stride_idx
            ):
                continue

            required_dim_order = _required_dim_order(
                value_stl,
                value_layout,
                value_coords,
                index_coords,
                access_subs,
                stride_idx,
            )
            required_stl = SpyreTensorLayout(
                value_layout.size,
                value_layout.stride,
                value_layout.dtype,
                required_dim_order,
            )

            if not is_own_output and _can_mutate_producer_in_place(
                graph, resolved_value_buf
            ):
                _rewrite_producer_layout(resolved_value_buf, required_stl)
            elif not is_own_output:
                required_layout = _fixed_tiled(value_layout, required_stl)
                op = _insert_relayout_copy(
                    graph, op, resolved_value_buf, required_layout
                )
            else:
                raise Unsupported(
                    f"enforce_indirect_layout: scatter output {op.get_name()!r} "
                    f"has a non-compliant indirect write layout; in-place "
                    f"mutation targets cannot be relayout-copied"
                )
