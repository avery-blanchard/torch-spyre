from torch._inductor.codegen.memory_planning import (
    Allocation,
    AllocationPool,
    AllocationPools,
    BufferGroup,
    MemoryPlanner,
    TemporalSplit,
    AllocFromPoolLine,
    DeallocFromPoolLine,
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

from torch_spyre._inductor.constants import SEGMENT_OFFSETS


def get_pool_id(pool_name: str) -> int:
    """Extract the pool ID number from a pool name (e.g., 'pool0' -> 0)."""
    return int(pool_name.removeprefix("pool"))


class SpyreAllocation(Allocation):
    """Spyre-specific allocation with HBM segment assignment."""

    def __post_init__(self):
        super().__post_init__()
        self.is_output = False

    def finalize(self, pool, offset):
        """Assign HBM address based on pool ID and number of graph inputs."""
        assert self.pool is None and self.offset is None
        self.pool = pool
        self.offset = offset

        # Map pool ID to HBM segment: inputs get segments 0..n-1, then pools start at segment n.
        n = SpyreAllocationPools._num_graph_inputs
        pool_id = get_pool_id(self.pool.name)
        segment_id = n + pool_id
        self.node.get_layout().allocation["hbm"] = SEGMENT_OFFSETS[segment_id] + int(
            self.offset
        )
        print("finalized allocation", self.node.get_layout().allocation["hbm"])
        return self

    def codegen_alloc_from_pool(self, wrapper):
        """Generate allocation code from pool, with special handling for outputs."""
        return "", []


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

    def allocate_output(self, block: Allocation):
        """Each output gets its own pool so its HBM address starts at offset 0."""
        print("IN ALLOCATE OUTPUT")
        pools = self.get_pools(block)
        block.mark_allocated()
        block.is_output = True
        pools.append(
            AllocationPool(
                block.device,
                TemporalSplit([block]),
                can_expand=False,
            )
        )

    def finalize(self):
        """Assign inputs to segments 0..n-1, then finalize pools."""
        for i, inp in enumerate(self._graph_inputs):
            inp.get_layout().allocation["hbm"] = SEGMENT_OFFSETS[i]
        super().finalize()


class SpyreAllocationPool(AllocationPool):
    def __post_init__(self) -> None:
        for block in self.root.allocations:
            if isinstance(block, SpyreAllocation):
                self.update_restrict_live_range(block)

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

    def codegen_create(self, wrapper, code: IndentedBuffer):
        assert self.name
        nbytes = self.root.get_symbolic_size()
        for block in self.root.allocations:
            if (
                isinstance(block, SpyreAllocation)
                and nbytes == block.get_symbolic_size()
            ):
                node = block.node
                code.writeline(
                    wrapper.make_allocation(
                        self.name,
                        device=self.device,
                        dtype=node.get_dtype(),
                        shape=tuple(node.get_size()),
                        stride=tuple(node.get_stride()),
                    )
                )
                return
        else:
            code.writeline(
                wrapper.make_allocation(
                    self.name,
                    device=self.device,
                    dtype=torch.uint8,
                    shape=(nbytes,),
                    stride=(1,),
                )
            )


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
