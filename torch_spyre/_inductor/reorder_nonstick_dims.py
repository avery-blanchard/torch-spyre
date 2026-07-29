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
coordinate access patterns.

This pass runs after insert_restickify, once every op has a committed
FixedTiledLayout. For operations with nonstick-dim-order requirements, it checks
whether the value tensor's current dim_order matches; if not, either rewrites the
producer's output layout in place (if the producer is a ComputedBuffer and not a
graph output) or inserts a spyre.restickify copy in the required layout. New
requirement sources can be added by extending _get_nonstick_dim_order_requirements().
"""

import sympy

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    ReinterpretView,
    Scatter,
    StorageBox,
)
from torch_spyre._C import SpyreTensorLayout

from .constants import ELIDED_COPY_BACK_ATTR
from .insert_restickify import (
    _create_restickify_node,
    _fixed_tiled,
    insert_restickify_on_node_inputs,
)
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .op_spec import IndirectAccess
from .pass_utils import (
    device_coordinates,
    indirect_info_from_op,
    _find_scatter_index_buf_names,
    _build_indirect_store_subs,
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


def _build_required_stl(
    value_stl: SpyreTensorLayout,
    indirect_device_pos: int,
) -> SpyreTensorLayout:
    """Build a new STL with the indirect coordinate rotated to device position 0.

    Takes the current device layout and rotates it so the indirect coordinate
    (at indirect_device_pos) moves to position 0, while keeping the stick
    (at position -1) at the end. Returns a new STL with the rotated layout.
    """
    device_size = list(value_stl.device_size)
    stride_map = list(value_stl.stride_map)
    n = len(device_size)
    stick_pos = n - 1

    # If indirect is already at position 0, no change needed
    if indirect_device_pos == 0:
        return value_stl

    # Rotate: move indirect_device_pos to position 0, keep stick at end
    order = (
        [indirect_device_pos]
        + [i for i in range(n) if i != indirect_device_pos and i != stick_pos]
        + [stick_pos]
    )

    new_device_size = [device_size[i] for i in order]
    new_stride_map = [stride_map[i] for i in order]

    return SpyreTensorLayout(
        device_size=new_device_size,
        stride_map=new_stride_map,
        device_dtype=value_stl.device_dtype,
    )


def _can_mutate_producer_in_place(value_buf, output_names: set[str]) -> bool:
    """Check if a value buffer's producer layout can be rewritten in place.

    Producer layout can be rewritten if the buffer is a ComputedBuffer (not
    a graph input), not a mutation layout, and not a graph output. Multiple
    consumers are fine — we're rewriting the producer's output, which all
    consumers will see.
    """
    if not isinstance(value_buf, ComputedBuffer):
        return False
    if isinstance(value_buf.layout, MutationLayoutSHOULDREMOVE):
        return False
    if value_buf.get_name() in output_names:
        return False
    return True


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


def _output_real_layout(op: ComputedBuffer) -> FixedTiledLayout:
    """Resolve an op's committed output layout, unwrapping a genuine mutation
    target (unlike _real_layout, which asserts mutation layouts only appear on
    elided copy-backs — an op's own MutationLayoutSHOULDREMOVE is expected)."""
    layout = op.get_layout()
    if isinstance(layout, MutationLayoutSHOULDREMOVE):
        layout = layout.real_layout()
    return layout


def _output_value_buf_for_op(
    graph: GraphLowering,
    op: ComputedBuffer,
    access_subs: dict,
    sizes: dict | None,
) -> ComputedBuffer | None:
    """Return op itself if it indirectly writes its own output (scatter: the
    write dep whose device_coordinates contain an IndirectAccess), else None.

    For Scatter ops, we need to check the scatter source (value tensor) layout,
    not the output layout (which is the mutation target with fixed coordinates).
    """

    write_dep = next(
        (d for d in op.get_read_writes().writes if isinstance(d, MemoryDep)), None
    )
    if write_dep is None:
        logger.debug("_output_value_buf_for_op %s: no write_dep", op.get_name())
        return None

    # For Scatter ops, check if there are scatter index buffers (indirect write indicator)
    if isinstance(op.data, Scatter):
        scatter_index_names = _find_scatter_index_buf_names(op)
        if scatter_index_names:
            logger.debug(
                "_output_value_buf_for_op %s: scatter with index buffers %s",
                op.get_name(),
                scatter_index_names,
            )
            return op
        logger.debug(
            "_output_value_buf_for_op %s: scatter with no index buffers", op.get_name()
        )
        return None

    # For non-Scatter ops, check write coordinates for IndirectAccess
    layout = _output_real_layout(op)
    if not isinstance(layout, FixedTiledLayout):
        return None
    coords = device_coordinates(layout.device_layout, write_dep, sizes)
    coords_substituted = [c.xreplace(access_subs) if access_subs else c for c in coords]
    if any(hasattr(c, "has") and c.has(IndirectAccess) for c in coords_substituted):
        return op
    return None


def _resolve_mutation_target(op: ComputedBuffer) -> tuple[str, object]:
    """Unwrap a MutationLayoutSHOULDREMOVE op's layout to (target_name, target_buf).

    Mirrors propagate_layouts.py's MutationLayoutSHOULDREMOVE branch, which
    unwraps ReinterpretView chains the same way to resolve the mutation target.
    """
    assert isinstance(op.layout, MutationLayoutSHOULDREMOVE)
    target = op.layout.target
    while isinstance(target, ReinterpretView):
        target = target.data
    return target.get_name(), target


def _insert_mutation_relayout_copy(
    graph: GraphLowering,
    mutation_op: ComputedBuffer,
    write_dep: MemoryDep,
    access_subs: dict,
    sizes: dict | None,
) -> None:
    """Fix a non-compliant indirect-write layout on a MutationLayoutSHOULDREMOVE op.

    Mirrors insert_post_mutation_restickify's copy-in / retarget / copy-back
    shape, built eagerly here (rather than via the _restickify_plan deferred
    queue, which insert_post_mutation_restickify alone owns) since
    reorder_nonstick_dims runs after every op already has a committed
    FixedTiledLayout.

    If a _restickify_plan is already pending on mutation_op (propagate_layouts
    staged a stick-offset fix that insert_post_mutation_restickify will apply
    later), don't build a second copy-in/copy-back pair. Instead, rotate the
    pending alt_stl so the single later copy already lands the target in a
    layout that satisfies both requirements. alt_stl lives in its own
    coordinate space (propagate_layouts.py's _candidate_output_stls builds it
    from the target's host size/stride with a possibly different dim chosen as
    the stick dim, not by permuting the current device stride_map), so the
    rotation is re-derived from scratch against alt_stl via the same
    indirect-position lookup used everywhere else in this pass, rather than by
    transplanting the independent branch's rotation order onto it.
    """
    pending_plan = getattr(mutation_op, "_restickify_plan", None)
    if pending_plan is not None:
        target_name, orig_stl, alt_stl = pending_plan
        alt_coords = device_coordinates(alt_stl, write_dep, sizes)
        alt_stride_idx = _indirect_stride_idx(alt_coords, access_subs)
        if alt_stride_idx is not None and not _dim_order_is_compliant(
            alt_stl, alt_stride_idx
        ):
            alt_indirect_pos = len(alt_stl.stride_map) - 1 - alt_stride_idx
            rotated_alt_stl = _build_required_stl(alt_stl, alt_indirect_pos)
        else:
            rotated_alt_stl = alt_stl
        mutation_op._restickify_plan = (target_name, orig_stl, rotated_alt_stl)
        graph_input = graph.graph_inputs.get(target_name)
        if graph_input is not None:
            graph_input.layouts = [rotated_alt_stl]
        logger.info(
            "enforce_indirect_layout: composed with pending restickify_plan for "
            "%s -> rotated alt_stl %s",
            target_name,
            list(rotated_alt_stl.stride_map),
        )
        return

    output_stl = _output_real_layout(mutation_op).device_layout
    write_stride_idx = _indirect_stride_idx(
        device_coordinates(output_stl, write_dep, sizes), access_subs
    )
    assert write_stride_idx is not None, (
        f"expected an IndirectAccess write coordinate on {mutation_op.get_name()!r}"
    )
    output_indirect_pos = len(output_stl.stride_map) - 1 - write_stride_idx
    required_stl = _build_required_stl(output_stl, output_indirect_pos)

    target_name, target_buf = _resolve_mutation_target(mutation_op)
    target_layout = target_buf.get_layout()
    assert isinstance(target_layout, FixedTiledLayout), (
        f"expected FixedTiledLayout on mutation target {target_name!r}, "
        f"got {type(target_layout).__name__}"
    )
    buf_tmp_layout = _fixed_tiled(target_layout, required_stl)
    orig_stl_layout = target_layout

    # Step 1: copy-in: target (current layout) -> buf_tmp (required_stl).
    _, buf_tmp = _create_restickify_node(
        {"arg_name": target_name, "target_layout": buf_tmp_layout}, mutation_op
    )
    buf_tmp_name = buf_tmp.get_name()
    buf_tmp._input_layout_overrides = {target_name: orig_stl_layout}

    # Step 2: retarget the mutation to buf_tmp, preserving any slice offset.
    mutation_name = mutation_op.get_name()
    original_layout = mutation_op.layout
    assert isinstance(original_layout, MutationLayoutSHOULDREMOVE)
    slice_layout = original_layout.target.get_layout()
    if isinstance(original_layout.target, ReinterpretView) and slice_layout.offset != 0:
        mutation_op.layout = MutationLayoutSHOULDREMOVE(
            ReinterpretView(data=StorageBox(buf_tmp), layout=slice_layout)
        )
    else:
        mutation_op.layout = MutationLayoutSHOULDREMOVE(buf_tmp)

    # Step 3: copy-back: buf_tmp (required_stl) -> target_buf (required_stl).
    buf_copyback_layout = _fixed_tiled(target_layout, required_stl)
    _, buf_copyback = _create_restickify_node(
        {"arg_name": buf_tmp_name, "target_layout": buf_copyback_layout},
        mutation_op,
    )
    buf_copyback.layout = MutationLayoutSHOULDREMOVE(target_buf)

    mutation_op._emit_set_layout = (target_name, required_stl)

    operations = graph.operations
    mutation_op_index = operations.index(mutation_op)
    operations.remove(buf_tmp)
    operations.insert(mutation_op_index, buf_tmp)
    operations.remove(buf_copyback)
    operations.insert(mutation_op_index + 2, buf_copyback)

    logger.info(
        "enforce_indirect_layout: inserted mutation relayout copy for %s "
        "(copy-in %s -> %s, copy-back %s -> %s)",
        mutation_name,
        target_name,
        buf_tmp_name,
        buf_tmp_name,
        target_name,
    )


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
        logger.debug(
            "reorder_nonstick_dims: checking op %s, is_scatter=%s, is_mutation=%s",
            original_op.get_name(),
            isinstance(original_op.data, Scatter),
            isinstance(original_op.layout, MutationLayoutSHOULDREMOVE),
        )
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

            # Rotate the device layout to put the indirect coordinate at position 0
            indirect_device_pos = len(value_stl.stride_map) - 1 - stride_idx
            required_stl = _build_required_stl(value_stl, indirect_device_pos)

            if _can_mutate_producer_in_place(value_buf, graph.get_output_names()):
                _rewrite_producer_layout(value_buf, required_stl)
            else:
                required_layout = _fixed_tiled(value_layout, required_stl)
                op = _insert_relayout_copy(graph, op, value_buf, required_layout)

        # For scatter ops, ensure scatter destination dimension 0 (scatter index) is outermost
        if isinstance(op.data, Scatter) and isinstance(
            op.layout, MutationLayoutSHOULDREMOVE
        ):
            write_dep = next(
                (d for d in op.get_read_writes().writes if isinstance(d, MemoryDep)),
                None,
            )
            if write_dep is not None:
                output_layout = _output_real_layout(op)
                if isinstance(output_layout, FixedTiledLayout):
                    output_stl = output_layout.device_layout
                    # For scatter, dimension 0 is the scatter index and must be outermost.
                    # Check if dimension 0 corresponds to stride_idx = len(stride_map)-1
                    required_stride_idx = len(output_stl.stride_map) - 1
                    is_compliant = _dim_order_is_compliant(
                        output_stl, required_stride_idx
                    )
                    logger.debug(
                        "scatter_destination_check: output dim 0 compliant=%s",
                        is_compliant,
                    )
                    if not is_compliant:
                        logger.info(
                            "scatter_destination_check: inserting mutation relayout copy for %s",
                            op.get_name(),
                        )
                        store_subs, sizes = _build_indirect_store_subs(op)
                        if store_subs:
                            scatter_access_subs = {
                                sym: IndirectAccess(sympy.Symbol(expr.base.name))
                                for sym, expr in store_subs.items()
                            }
                            _insert_mutation_relayout_copy(
                                graph,
                                op,
                                write_dep,
                                scatter_access_subs,
                                sizes,
                            )
