import math
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
        layout = self.node.get_layout()
        device_size = layout.device_layout.device_size
        # Compute the byte size of this tensor on device: the product of all
        # non-stick dimensions times 128 bytes per stick.
        device_nbytes = math.prod(device_size[:-1]) * 128
        self.offset = (offset // self.symbolic_size) * device_nbytes
        self.node.get_layout().allocation["hbm"] = SEGMENT_OFFSETS[-1] + int(
            self.offset
        )
        return self

    def codegen_alloc_from_pool(self, wrapper):
        return "", []


class SpyreTemporalSplit(TemporalSplit):
    def finalize(self, pool, offset):
        current_offset = offset
        finalized_allocations = []
        for block in self.allocations:
            finalized_block = block.finalize(pool, current_offset)
            finalized_allocations.append(finalized_block)
            current_offset += align(finalized_block.get_symbolic_size())
        self.allocations = finalized_allocations
        self.clear_cache()
        if len(self.allocations) == 1:
            return self.allocations[0]
        return self

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
                print("Creating SpyreSpatialSplit with offset:", slot_size - block_size)
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
                print("Growing allocation", self.allocations)
            return True


class SpyreSpatialSplit(SpatialSplit):
    left: SpyreTemporalSplit
    right: SpyreTemporalSplit

    @staticmethod
    def create(left, extra_space):
        print("In SpyreSpatialSplit.create")
        assert isinstance(left, AllocationTreeNode)
        assert isinstance(extra_space, int) and extra_space >= 1
        return SpyreSpatialSplit(
            SpyreTemporalSplit([left]), SpyreTemporalSplit([Empty(extra_space)])
        )

    def _allocate(self, block: SpyreAllocation, is_last: bool):
        print("In SpyreSpatialSplit.allocate")
        return self.left.allocate(block, False) or self.right.allocate(block, is_last)

    def finalize(self, pool, offset):
        print("In SpyreSpatialSplit.finalize left", offset)
        print(
            "In SpyreSpatialSplit.finalize right", align(self.left.get_symbolic_size())
        )
        left_size = self.left.get_symbolic_size()
        self.left = self.left.finalize(pool, offset)
        self.right = self.right.finalize(pool, offset + align(left_size))
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
    """Spyre-specific allocation pools that assign inputs/outputs to distinct segments."""

    _graph_inputs = list(V.graph.graph_inputs.values())
    _num_graph_inputs = len(list(V.graph.graph_inputs.values()))
    _graph_outputs = V.graph.graph_outputs
    _graph_input_names = list(V.graph.graph_input_names)
    _graph_output_names = list(V.graph.get_output_names())

    def allocate(self, block: SpyreAllocation):
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
            # First allocation: create the pool
            pools.append(
                SpyreAllocationPool(
                    block.device,
                    SpyreTemporalSplit([block]),
                    can_expand=True,
                )
            )
            block.mark_allocated()
            return

    def finalize(self):
        mutation_real_name = V.graph.scheduler.mutation_real_name
        num_inputs = len(self._graph_input_names)
        for i, name in enumerate(self._graph_input_names):
            V.graph.get_buffer(name).get_layout().allocation["hbm"] = SEGMENT_OFFSETS[i]
        for i, name in enumerate(self._graph_output_names):
            real_name = mutation_real_name.get(name, name)
            V.graph.get_buffer(real_name).get_layout().allocation["hbm"] = (
                SEGMENT_OFFSETS[i + num_inputs]
            )

        super().finalize()


class SpyreAllocationPool(AllocationPool):
    def __post_init__(self) -> None:
        for block in self.root.allocations:
            if isinstance(block, SpyreAllocation):
                self.update_restrict_live_range(block)

    def allocate_at_end(self, block: SpyreAllocation) -> bool:
        """Override to use SpyreSpatialSplit instead of SpatialSplit."""
        print("allocate at end")
        block.mark_allocated()
        self.root = SpyreTemporalSplit(
            [SpyreSpatialSplit(self.root, SpyreTemporalSplit([block]))]
        )
        return True

    def allocate(self, block: SpyreAllocation, is_last: bool):
        if (
            self.restrict_live_range is not None
            and not self.restrict_live_range.contains(block.live_range)
        ):
            return False

        # block_earliest_available = block.get_earliest_available()
        # pool_begin = self.root.get_live_ranges().begin
        # if block_earliest_available and block_earliest_available > pool_begin:
        #     return False

        is_last = self.can_expand and is_last
        if self.root.allocate(block, is_last):
            # self.update_restrict_live_range(block)
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
        """Call all the memory planning passes, then clean up intermediate buffers."""
        lines = super().plan(lines)
        # Determine and populate intermediate buffers
        self.determine_intermediate_buffers(lines)
        # Remove lines referencing intermediate buffers
        self.remove_intermediate_buffers(lines)
        return lines

    def determine_intermediate_buffers(self, lines):
        for line in lines:
            if isinstance(line, SpyreAllocFromPoolLine):
                allocation = line.group.allocation
                if allocation and not allocation.is_output:
                    for buf_name in line.group.names:
                        self.wrapper.intermediate_buffers.add(buf_name)

    def remove_intermediate_buffers(self, lines):
        intermediates = self.wrapper.intermediate_buffers
        if not intermediates:
            return

        filtered_lines = []
        for line in lines:
            # Skip lines that only operate on intermediate buffers
            if hasattr(line, "node") and line.node.get_name() in intermediates:
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
