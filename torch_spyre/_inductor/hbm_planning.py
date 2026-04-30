# Copyright 2026 The Torch-Spyre Authors.
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

import math

from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.virtualized import V

from .constants import HBM_INTERMEDIATES_POOL_BASE
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger

logger = get_inductor_logger("HBM_PLANNING")

# HBM sticks are 128 bytes; all allocations are stick-aligned.
_STICK_BYTES = 128


class SpyreHBMAllocator:
    """Greedy temporal-reuse allocator for the HBM intermediate tensor pool.

    Tracks a set of free blocks within a single contiguous HBM region. Buffers
    whose live ranges do not overlap share the same region. Each block is a
    (offset, size) pair measured in bytes from HBM_INTERMEDIATES_POOL_BASE.
    """

    def __init__(self) -> None:
        self._free: list[tuple[int, int]] = []  # (offset, size) free blocks
        self._pool_end: int = 0  # current end of the pool

    def allocate(self, size: int) -> int:
        """Return a byte offset from HBM_INTERMEDIATES_POOL_BASE for a block of
        `size` bytes. Reuses an existing free block when possible."""
        for i, (blk_offset, blk_size) in enumerate(self._free):
            if blk_size >= size:
                self._free.pop(i)
                # Return any leftover fragment to the free list.
                remainder = blk_size - size
                if remainder > 0:
                    self._free.append((blk_offset + size, remainder))
                return blk_offset
        # No suitable free block — extend the pool.
        offset = self._pool_end
        self._pool_end += size
        return offset

    def free(self, offset: int, size: int) -> None:
        """Return a previously allocated block to the free list."""
        self._free.append((offset, size))


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


def _compute_size_bytes(name: str) -> int:
    """Return the stick-aligned device size in bytes for buffer `name`."""
    buf = V.graph.get_buffer(name)
    layout = buf.get_layout()
    assert isinstance(layout, FixedTiledLayout), (
        f"hbm_planning: expected FixedTiledLayout for {name}, got {type(layout)}"
    )
    dev_layout = layout.device_layout
    # device_size[-1] is the stick dimension (always 1 in the size array);
    # the product of all other dims is the number of sticks.
    num_sticks = math.prod(dev_layout.device_size[:-1])
    size_bytes = num_sticks * _STICK_BYTES
    return _align_up(size_bytes, _STICK_BYTES)


def _compute_live_ranges(
    nodes: list[BaseSchedulerNode],
    intermediates: set[str],
) -> dict[str, tuple[int, int]]:
    """Return {buf_name: (birth_step, death_step)} for each intermediate.

    birth_step: timestep of the node that writes the buffer.
    death_step: last timestep at which any node reads the buffer.
    Both are indices into `nodes`.
    """
    birth: dict[str, int] = {}
    death: dict[str, int] = {}

    for idx, node in enumerate(nodes):
        rw = node.read_writes
        for dep in rw.writes:
            if dep.name in intermediates:
                birth[dep.name] = idx
        for dep in rw.reads:
            if dep.name in intermediates:
                death[dep.name] = idx

    live_ranges: dict[str, tuple[int, int]] = {}
    for name in intermediates:
        if name in birth:
            live_ranges[name] = (birth[name], death.get(name, birth[name]))
    return live_ranges


def hbm_memory_planning(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """Assign pooled HBM addresses to intermediate tensors.

    Runs on the pre-fusion list[BaseSchedulerNode]. Identifies intermediate
    buffers (not graph inputs/outputs, not already LX-allocated), performs
    live range analysis, and assigns layout.allocation["hbm"] = address so
    that non-overlapping intermediates share HBM memory.
    """
    graph_inputs: set[str] = set(V.graph.graph_inputs.keys())
    graph_outputs: set[str] = set(V.graph.get_output_names())

    # Collect intermediate buffer names from all node write sets.
    intermediates: set[str] = set()
    for node in nodes:
        for dep in node.read_writes.writes:
            name = dep.name
            if name in graph_inputs or name in graph_outputs:
                continue
            buf = V.graph.get_buffer(name)
            if buf is None:
                continue
            layout = buf.get_layout()
            if not isinstance(layout, FixedTiledLayout):
                continue
            if layout.allocation:
                # Already assigned (e.g. LX scratchpad) — skip.
                continue
            intermediates.add(name)

    if not intermediates:
        return nodes

    live_ranges = _compute_live_ranges(nodes, intermediates)

    # Sort by birth step so the allocator processes tensors in execution order.
    sorted_bufs = sorted(live_ranges.items(), key=lambda kv: kv[1][0])

    allocator = SpyreHBMAllocator()
    # Track (death_step, offset, size) so we can free blocks promptly.
    pending_frees: list[tuple[int, int, int]] = []

    for name, (birth, death) in sorted_bufs:
        # Free any blocks whose live range ended before this birth step.
        still_live = []
        for entry in pending_frees:
            d, off, sz = entry
            if d < birth:
                allocator.free(off, sz)
            else:
                still_live.append(entry)
        pending_frees = still_live

        size = _compute_size_bytes(name)
        offset = allocator.allocate(size)
        address = HBM_INTERMEDIATES_POOL_BASE + offset

        layout = V.graph.get_buffer(name).get_layout()
        layout.allocation["hbm"] = address
        pending_frees.append((death, offset, size))

        logger.debug(
            "hbm_planning: %s  live=[%d,%d]  size=%d  addr=0x%x",
            name,
            birth,
            death,
            size,
            address,
        )

    return nodes
