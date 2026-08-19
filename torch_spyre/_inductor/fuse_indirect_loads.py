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

"""Fuse pure indirect-load ops into their sole consumer's inner_fn.

Pre-scheduling IR pass. For each Pointwise ComputedBuffer that is a pure
indirect load (bare x[i], no other ops) with exactly one IR-level consumer,
splices the gather's inner_fn directly into the consumer's inner_fn via a
WrapperHandler, and deletes the separate gather buffer from operations.

This is a generic solution that works for any operation consuming the gather,
not just those with custom lowerings.
"""

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Pointwise
from torch._inductor.ops_handler import WrapperHandler
from torch._inductor.virtualized import V

from .logging_utils import get_inductor_logger
from .pass_utils import replace_computed_buffer_body

logger = get_inductor_logger("fuse_indirect_loads")


def _is_pure_indirect_load(op: ComputedBuffer) -> bool:
    """Check if op is a bare indirect load (x[i]) with no other ops fused.

    Returns True iff:
    - op.data is a Pointwise (not a Reduction or other op type)
    - get_read_writes().reads contains exactly one MemoryDep
    - that MemoryDep is an indirect access (MemoryDep.is_indirect() == True)
    """
    if not isinstance(op, ComputedBuffer) or not isinstance(op.data, Pointwise):
        return False
    reads = op.get_read_writes().reads
    if len(reads) != 1:
        return False
    dep = next(iter(reads))
    if not isinstance(dep, MemoryDep):
        return False
    return dep.is_indirect()


def _find_ir_consumers(buf_name: str, operations: list) -> list[ComputedBuffer]:
    """Find all ComputedBuffer ops in operations that read buf_name.

    Scans each op's get_read_writes().reads for a MemoryDep with name==buf_name.
    """
    consumers = []
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        for dep in op.get_read_writes().reads:
            if isinstance(dep, MemoryDep) and dep.name == buf_name:
                consumers.append(op)
                break
    return consumers


class _GatherSpliceHandler(WrapperHandler):
    """Splice a gather's inner_fn in place of ops.load(gather_buf_name, ...).

    When the consumer's inner_fn calls ops.load(gather_buf_name, index),
    instead call the gather's own inner_fn directly, which will execute the
    gather's own indirect_indexing() and load() calls within the consumer's
    iteration-space symbols.
    """

    def __init__(
        self,
        inner,
        gather_buf_name: str,
        gather_inner_fn,
        gather_inner_fn_args_fn,
    ):
        super().__init__(inner)
        self._gather_buf_name = gather_buf_name
        self._gather_inner_fn = gather_inner_fn
        self._gather_inner_fn_args_fn = gather_inner_fn_args_fn

    def load(self, name, index):
        if name == self._gather_buf_name:
            return self._gather_inner_fn(*self._gather_inner_fn_args_fn())
        return super().load(name, index)


def _fuse_gather_into_consumer(
    gather_op: ComputedBuffer,
    consumer_op: ComputedBuffer,
    operations: list,
) -> None:
    """Splice gather's inner_fn into consumer's, delete gather from operations.

    Patches consumer_op.data.inner_fn to call gather_op's inner_fn when
    loading gather_op's buffer, then reconstructs the consumer to invalidate
    caches, and removes gather_op from operations.
    """
    gather_name = gather_op.get_name()
    gather_inner_fn = gather_op.data.inner_fn
    gather_inner_fn_args = gather_op.data.inner_fn_args

    logger.debug(
        "fuse_indirect_loads: fusing gather %s into consumer %s",
        gather_name,
        consumer_op.get_name(),
    )

    orig_consumer_inner = consumer_op.data.inner_fn

    def new_inner_fn(
        *args,
        _gather_name=gather_name,
        _gather_inner=gather_inner_fn,
        _gather_args=gather_inner_fn_args,
        _orig=orig_consumer_inner,
    ):
        handler = _GatherSpliceHandler(V.ops, _gather_name, _gather_inner, _gather_args)
        with V.set_ops_handler(handler):
            return _orig(*args)

    object.__setattr__(consumer_op.data, "inner_fn", new_inner_fn)

    replace_computed_buffer_body(
        consumer_op,
        consumer_op.data,
        operations,
        pass_name="fuse_indirect_loads",
        reason="fuse indirect load into consumer inner_fn",
    )

    idx = operations.index(gather_op)
    operations.pop(idx)

    logger.debug(
        "fuse_indirect_loads: removed gather op %s from operations", gather_name
    )


def fuse_indirect_loads(graph: GraphLowering) -> None:
    """Main entry point: fuse pure indirect-load ops into their sole consumer.

    Runs pre-scheduling, after enforce_indirect_access_layout but before
    scheduler construction. Scans graph.operations for pure indirect-load
    Pointwise ComputedBuffers; for each with exactly one IR-level consumer,
    splices it into that consumer's inner_fn and removes it from operations.
    """
    gathers_to_fuse = []
    for op in list(graph.operations):
        if _is_pure_indirect_load(op):
            gathers_to_fuse.append(op)

    if not gathers_to_fuse:
        logger.debug("fuse_indirect_loads: no pure indirect loads found")
        return

    logger.debug(
        "fuse_indirect_loads: found %d pure indirect load ops",
        len(gathers_to_fuse),
    )

    for gather_op in gathers_to_fuse:
        gather_name = gather_op.get_name()
        consumers = _find_ir_consumers(gather_name, graph.operations)

        if len(consumers) == 0:
            logger.debug(
                "fuse_indirect_loads: gather %s has no IR consumers, skipping",
                gather_name,
            )
            continue
        if len(consumers) > 1:
            logger.debug(
                "fuse_indirect_loads: gather %s has %d consumers, not fusing",
                gather_name,
                len(consumers),
            )
            continue

        consumer_op = consumers[0]
        _fuse_gather_into_consumer(gather_op, consumer_op, graph.operations)
