from torch._inductor.codegen.memory_planning import (
    Allocation,
    AllocationPool,
    AllocationPools,
    AllocationTreeNode,
    BufferGroup,
    MemoryPlanner,
    Empty,
    SpatialSplit,
    TemporalSplit,
    AllocFromPoolLine,
    DeallocFromPoolLine,
    align,
)
from torch._inductor.codegen.wrapper import (
    AllocateLine,
    FreeIfNotReusedLine,
    NullLine,
    ReuseLine,
    IndentedBuffer,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from torch_spyre._inductor.constants import SEGMENT_OFFSETS, SEGMENT_SIZE


class SpyreAllocation(Allocation):
    """Spyre-specific allocation with HBM segment assignment."""

    def __post_init__(self):
        super().__post_init__()
        self.is_output = False

    def finalize(self, pool, offset):
        assert self.pool is None and self.offset is None
        self.pool = pool
        self.offset = offset
        print("Offset in SpyreAllocation.finalize", int(offset))
        segment_id = -1
        self.node.get_layout().allocation["hbm"] = SEGMENT_OFFSETS[segment_id] + int(
            self.offset
        )
        return self

    def codegen_alloc_from_pool(self, wrapper):
        return "", []


class SpyreTemporalSplit(TemporalSplit):
    def _allocate(self, block: SpyreAllocation, is_last: bool):
        slot_size = self.get_size_hint()
        block_size = block.get_size_hint()
        if not is_last and block_size > slot_size:
            return False  # doesn't fit

        block_live = block.get_live_ranges()
        overlapping = [
            s for s in self.allocations if s.get_live_ranges().overlaps(block_live)
        ]
        if len(overlapping) > 1:
            return False
        elif len(overlapping) == 1:
            return overlapping[0].allocate(block, is_last)
        else:
            block.mark_allocated()

            if len(self.allocations) == 1 and isinstance(self.allocations[-1], Empty):
                self.allocations.pop()

            if slot_size == block_size:
                self.allocations.append(block)
            elif slot_size > block_size:
                self.allocations.append(
                    SpyreSpatialSplit.create(block, slot_size - block_size)
                )
            else:  # grow this allocation
                assert is_last
                self.allocations = [
                    *(
                        SpyreSpatialSplit.create(a, block_size - slot_size)
                        for a in self.allocations
                    ),
                    block,
                ]
            return True


class SpyreSpatialSplit(SpatialSplit):
    left: SpyreTemporalSplit
    right: SpyreTemporalSplit

    @staticmethod
    def create(left, extra_space):
        assert isinstance(left, AllocationTreeNode)
        assert isinstance(extra_space, int) and extra_space >= 1
        return SpyreSpatialSplit(
            SpyreTemporalSplit([left]), SpyreTemporalSplit([Empty(extra_space)])
        )

    def finalize(self, pool, offset):
        self.left = self.left.finalize(pool, offset)
        self.right = self.right.finalize(
            pool, offset + align(self.left.get_symbolic_size())
        )
        self.clear_cache()
        if self.right.is_empty():
            return self.left
        return self


class SpyreAllocFromPoolLine(AllocFromPoolLine):
    """Similar to AllocationLine, but takes memory from a pool"""

    is_first_pool_usage: bool = False

    def codegen(self, code: IndentedBuffer):
        allocation = self.group.allocation
        assert allocation and allocation.pool
        pool = allocation.pool
        name = self.node.get_name()

        if self.is_first_pool_usage:
            pool.codegen_create(self.wrapper, code)

        # pool.names_to_del.extend(self.group.names)
        alloc_from_pool, allocation_lines_to_write = allocation.codegen_alloc_from_pool(
            self.wrapper
        )
        code.writelines(allocation_lines_to_write)
        if alloc_from_pool:
            if alloc_from_pool in pool.creation_cache:
                code.writeline(
                    self.wrapper.make_tensor_alias(
                        name, pool.creation_cache[alloc_from_pool], "alloc"
                    )
                )
            else:
                pool.creation_cache[alloc_from_pool] = name
                code.writeline(
                    f"{self.wrapper.declare}{name} = {alloc_from_pool}{self.wrapper.ending}"
                )


class SpyreAllocationPools(AllocationPools):
    """Spyre-specific allocation pools that assign inputs to distinct segments."""

    _graph_inputs = list(V.graph.graph_inputs.values())
    _num_graph_inputs = len(list(V.graph.graph_inputs.values()))
    _graph_outputs = V.graph.graph_outputs

    def allocate(self, block: Allocation):
        """Allocate into the single intermediate pool, enforcing MAX_POOLS and SEGMENT_SIZE."""
        pools = self.get_pools(block)

        if pools:
            pool = pools[0]
            # Reject if adding this block would exceed one segment's worth of memory.
            current = pool.root.get_symbolic_size()
            incoming = block.symbolic_size
            if current + incoming <= SEGMENT_SIZE:
                if pool.allocate(block, is_last=True):
                    return
            else:
                raise RuntimeError(
                    f"Intermediate pool would exceed the {SEGMENT_SIZE:#x}-byte "
                    "segment size limit. Only one pool is supported."
                )
        else:
            # First allocation: create the single pool.
            pools.append(
                AllocationPool(
                    block.device,
                    SpyreTemporalSplit([block]),
                    can_expand=True,
                )
            )
            block.mark_allocated()
            return

    def finalize(self):
        """Assign inputs/outputs to segments 0..n-1, then finalize pools."""
        for i, inp in enumerate(self._graph_inputs):
            inp.get_layout().allocation["hbm"] = SEGMENT_OFFSETS[i]
        for i, inp in enumerate(self._graph_outputs):
            inp.get_layout().allocation["hbm"] = SEGMENT_OFFSETS[
                i + self._num_graph_inputs
            ]
        super().finalize()


class SpyreAllocationPool(AllocationPool):
    def __post_init__(self) -> None:
        for block in self.root.allocations:
            if isinstance(block, SpyreAllocation):
                self.update_restrict_live_range(block)

    def allocate_at_end(self, block: Allocation) -> bool:
        """Override to use SpyreSpatialSplit instead of SpatialSplit."""
        block.mark_allocated()
        self.root = SpyreTemporalSplit(
            [SpyreSpatialSplit(self.root, SpyreTemporalSplit([block]))]
        )
        self.update_restrict_live_range(block)
        return True

    def allocate(self, block: SpyreAllocation, is_last: bool):
        if (
            self.restrict_live_range is not None
            and not self.restrict_live_range.contains(block.live_range)
        ):
            return False

        block_earliest_available = block.get_earliest_available()
        pool_begin = self.root.get_live_ranges().begin
        if block_earliest_available and block_earliest_available > pool_begin:
            return False

        is_last = self.can_expand and is_last
        if self.root.allocate(block, is_last):
            self.update_restrict_live_range(block)
            return True

        if is_last:
            return self.allocate_at_end(block)

        return False


class SpyreBufferGroup(BufferGroup):
    """Spyre-specific BufferGroup that creates SpyreAllocation objects."""

    def make_allocation(self):
        """Create a SpyreAllocation instead of upstream Allocation."""
        self.allocation = SpyreAllocation(
            self.node,
            self.live_range,
            size_hint=V.graph.sizevars.size_hint(self.sym_nbytes(), fallback=64),
            symbolic_size=self.sym_nbytes(),
        )


class SpyreMemoryPlanner(MemoryPlanner):
    """Spyre-specific memory planner using SpyreAllocationPools."""

    def __init__(self, wrapper):
        super().__init__(wrapper, pools=SpyreAllocationPools())

    def plan(self, lines):
        """Call all the memory planning passes in sequence, then clean up optimized buffers."""
        lines = super().plan(lines)
        # Determine and populate optimized-away buffers
        self.determine_optimized_buffers(lines)
        # Remove lines referencing optimized-away buffers
        self.remove_optimized_buffers(lines)
        return lines

    def determine_optimized_buffers(self, lines):
        """Determine which pool-allocated buffers will be optimized away (no code generation).

        A buffer is optimized away if it's converted to SpyreAllocFromPoolLine but
        its allocation.codegen_alloc_from_pool() returns empty string.
        """
        for line in lines:
            if isinstance(line, SpyreAllocFromPoolLine):
                allocation = line.group.allocation
                if allocation and not allocation.is_output:
                    for buf_name in line.group.names:
                        self.wrapper.intermediate_buffers.add(buf_name)

    def remove_optimized_buffers(self, lines):
        """Remove or neutralize lines that reference optimized-away buffers."""
        optimized_away = self.wrapper.intermediate_buffers
        if not optimized_away:
            return

        filtered_lines = []
        for line in lines:
            # Skip lines that only operate on optimized-away buffers
            if hasattr(line, "node") and line.node.get_name() in optimized_away:
                # Replace with NullLine to remove it from output
                filtered_lines.append(NullLine(self.wrapper))
            else:
                filtered_lines.append(line)

        lines[:] = filtered_lines

    def compute_buffer_groups(self, lines):
        """Populate buffer_groups with BufferGroup objects joining allocations with common storage."""
        name_to_group = {}
        for line in lines:
            if isinstance(line, AllocateLine):
                name = line.node.get_name()
                assert name not in name_to_group
                name_to_group[name] = SpyreBufferGroup(line.node)
            elif isinstance(line, ReuseLine):
                old_name = line.node.get_name()
                new_name = line.reused_as.get_name()
                assert new_name not in name_to_group
                if old_name in name_to_group:
                    name_to_group[old_name].names.append(new_name)
                    name_to_group[new_name] = name_to_group[old_name]

        outputs = OrderedSet(V.graph.get_output_names())
        unique_groups = [*{id(g): g for g in name_to_group.values()}.values()]
        for group in unique_groups:
            group.is_output = any(x in outputs for x in group.names)

        assert self.buffer_groups is None
        self.buffer_groups = unique_groups
        return name_to_group

    def convert_to_pool_lines(self, lines):
        name_to_group = self.compute_buffer_groups(lines)
        for i, line in enumerate(lines):
            if isinstance(line, AllocateLine):
                if line.node.get_name() in name_to_group:
                    group = name_to_group[line.node.get_name()]
                if not group.is_output:
                    lines[i] = SpyreAllocFromPoolLine(self.wrapper, group)
                elif isinstance(line, FreeIfNotReusedLine):
                    assert not line.is_reused
                    if line.node.get_name() in name_to_group:
                        group = name_to_group[line.node.get_name()]
                        if not group.is_output:
                            lines[i] = DeallocFromPoolLine(self.wrapper, group)
                        elif isinstance(line, ReuseLine):
                            if line.node.get_name() in name_to_group:
                                line.delete_old = False

    def allocate_groups(self):
        """Assign every allocation to a specific location in a specific AllocationPool."""
        assert self.buffer_groups is not None

        for group in self.buffer_groups:
            group.make_allocation()

        intermediates: list[Allocation] = []
        for group in self.buffer_groups:
            assert group.allocation
            if not group.is_output:
                intermediates.append(group.allocation)

        for block in sorted(
            intermediates, key=lambda x: (-x.size_hint, -len(x.live_range))
        ):
            self.pools.allocate(block)

        self.pools.finalize()
