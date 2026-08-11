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

"""Consolidated scatter-style indirect-access tests (one file per op family).

Each scenario routes its compile through
self._stage_and_e2e(...): it asserts across every capture-path stage --
classification, op-spec structure (IndirectAccess on the output), and SDSC
fields -- and then runs the kernel end-to-end on the real backend. The e2e run
reports an expected failure (pytest.xfail) on the value divergence / backend
abort the backend currently produces for indirect scatter, while the
capture-path checks above stay strict (a stage regression fails red).

All scatter scenarios run with SENCORES=1.

Status (validated on hardware build): index-tensor scatters reach a real op
spec with IndirectAccess on the output (SCATTER_OP_SPEC); the deeptools backend
diverges from / aborts on the bundle, surfaced here as xfail.
"""

import os
import sys

import torch
from torch.spyre import SpyreTensorLayout

sys.path.insert(0, os.path.dirname(__file__))
from indirect_access_common import (  # noqa: E402
    CRASHED,
    GATHER_OP_SPEC,
    SCATTER_OP_SPEC,
    DIRECT_OP_SPEC,
    IndirectAccessTestCase,
    canonical_device_layout,
)

from torch_spyre._inductor import config  # noqa: E402


@config.patch({"sencores": 1})
class TestScatter(IndirectAccessTestCase):
    """torch scatter-family ops: one compile + all-stage checks per scenario."""

    def _row_store(self, M=128, N=256, P=3, dtype=torch.int32):
        """Common row-store operands: out[M,N], src[P,N], 1-D idx[P], all named."""
        out = torch.zeros(M, N, dtype=torch.float16).to("spyre")
        src = torch.rand(P, N, dtype=torch.float16).to("spyre")
        idx = torch.randint(0, M, (P,), dtype=dtype).to("spyre")
        self.name_dims(out, {"M": M, "N": N})
        self.name_dims(src, {"P": P, "N": N})
        self.name_dims(idx, {"P": P})
        return out, src, idx

    def _full_index_store(self, M=128, N=256, P=3, dtype=torch.int32):
        """Operands for scatter with a full [P,N] index tensor: out[M,N], src[P,N]."""
        out = torch.zeros(M, N, dtype=torch.float16).to("spyre")
        src = torch.rand(P, N, dtype=torch.float16).to("spyre")
        index = torch.randint(0, M, (P, N), dtype=dtype).to("spyre")
        self.name_dims(out, {"M": M, "N": N})
        self.name_dims(src, {"P": P, "N": N})
        self.name_dims(index, {"P": P, "N": N})
        return out, src, index

    def _row_store_shape(self, M=128, N=256, P=3, dtype=torch.int32):
        """Shape-parameterized row-store operands, like _row_store but with a
        unique (non-repeating) 1-D index -- torch.randint can repeat a row,
        which is fine for accumulating ops but makes a non-accumulating
        scatter's CPU reference order-dependent. randperm guarantees distinct
        target rows so the e2e comparison is unambiguous regardless of P.

        `out` gets an explicit STL matching the canonical layout the compiler
        would pick anyway (the indexed row dim M is already outermost for a
        plain 2-D tensor) -- spelled out rather than implicit."""
        out_layout = canonical_device_layout((M, N), torch.float16)
        out = torch.zeros(M, N, dtype=torch.float16).to(device_layout=out_layout)
        src = torch.rand(P, N, dtype=torch.float16).to("spyre")
        idx = torch.randperm(M)[:P].to(dtype).to("spyre")
        self.name_dims(out, {"M": M, "N": N})
        self.name_dims(src, {"P": P, "N": N})
        self.name_dims(idx, {"P": P})
        return out, src, idx

    def _full_index_store_shape(self, M=128, N=256, P=3, dtype=torch.int32):
        """Shape-parameterized full-index-store operands, like
        _full_index_store but with a unique index per column (each column of
        `index` is an independent randperm) so a non-accumulating scatter's
        CPU reference does not depend on write order.

        `out` gets an explicit STL matching the canonical layout (see
        _row_store_shape)."""
        out_layout = canonical_device_layout((M, N), torch.float16)
        out = torch.zeros(M, N, dtype=torch.float16).to(device_layout=out_layout)
        src = torch.rand(P, N, dtype=torch.float16).to("spyre")
        index = torch.stack([torch.randperm(M)[:P] for _ in range(N)], dim=1).to(dtype)
        index = index.to("spyre")
        self.name_dims(out, {"M": M, "N": N})
        self.name_dims(src, {"P": P, "N": N})
        self.name_dims(index, {"P": P, "N": N})
        return out, src, index

    # -- Working index-tensor scatters: op spec with output IndirectAccess --
    def test_index_put(self):
        """out[idx] = src"""
        out, src, idx = self._row_store()

        def kernel(out, src, idx):
            out[idx] = src
            return out

        self._stage_and_e2e(kernel, out, src, idx, expect=SCATTER_OP_SPEC)

    def test_index_put_with_exp(self):
        """out[idx] = src.exp() -- index_put fused with a unary operation."""
        out, src, idx = self._row_store()

        def kernel(out, src, idx):
            out[idx] = src.exp()
            return out

        self._stage_and_e2e(kernel, out, src, idx, expect=SCATTER_OP_SPEC, op="exp")

    def test_scatter(self):
        """torch.scatter(out, 0, index, src)"""
        out, src, index = self._full_index_store(dtype=torch.int64)

        def kernel(out, src, index):
            return torch.scatter(out, 0, index, src)

        self._stage_and_e2e(kernel, out, src, index, expect=SCATTER_OP_SPEC)

    def test_scatter_method_without_unary(self):
        """out.scatter_(0, index, src) -- in-place method form without a unary."""
        out, src, index = self._full_index_store()

        def kernel(out, src, index):
            return out.scatter_(0, index, src)

        self._stage_and_e2e(kernel, out, src, index, expect=SCATTER_OP_SPEC)

    def test_scatter_with_exp(self):
        """y.scatter_(0, index, src.exp()) -- fused unary, exp runs on Spyre.

        Also pins the detection gap: indirect_info_from_op flags gather
        loads but not scatter stores (the output is recognized later in
        superdsc via is_output_tensor), so detected=False here.
        """
        out, src, index = self._full_index_store()

        def kernel(out, src, index):
            return out.scatter_(0, index, src.exp())

        self._stage_and_e2e(
            kernel,
            out,
            src,
            index,
            expect=SCATTER_OP_SPEC,
            op="exp",
            detected=False,
        )

    def test_scatter_add(self):
        """y.scatter_add_(0, index, src)"""
        out, src, index = self._full_index_store()

        def kernel(out, src, index):
            return out.scatter_add_(0, index, src)

        self._stage_and_e2e(kernel, out, src, index, expect=SCATTER_OP_SPEC)

    def test_index_copy(self):
        """torch.index_copy(out, 0, idx, src).

        index_copy requires a long (int64) index, unlike the int32-friendly
        index_put/index_add, so the CPU reference needs an int64 index here.
        """
        out, src, idx = self._row_store(dtype=torch.int64)

        def kernel(out, src, idx):
            return torch.index_copy(out, 0, idx, src)

        self._stage_and_e2e(kernel, out, src, idx, expect=SCATTER_OP_SPEC)

    def test_index_add(self):
        """out.index_add_(0, idx, src)"""
        out, src, idx = self._row_store()

        def kernel(out, src, idx):
            return out.index_add_(0, idx, src)

        self._stage_and_e2e(kernel, out, src, idx, expect=SCATTER_OP_SPEC)

    def test_scatter_reduce(self):
        """out.scatter_reduce_(0, index, src, "sum")"""
        out, src, index = self._full_index_store(dtype=torch.int64)

        def kernel(out, src, index):
            return out.scatter_reduce_(0, index, src, "sum")

        self._stage_and_e2e(kernel, out, src, index, expect=SCATTER_OP_SPEC)

    def test_index_put_accumulate(self):
        """out.index_put_((idx,), src, accumulate=True) -- out[idx] += src."""
        out, src, idx = self._row_store()

        def kernel(out, src, idx):
            return out.index_put_((idx,), src, accumulate=True)

        self._stage_and_e2e(kernel, out, src, idx, expect=SCATTER_OP_SPEC)

    def test_scatter_add_functional(self):
        """torch.scatter_add(out, 0, index, src) -- functional accumulating scatter."""
        out, src, index = self._full_index_store()

        def kernel(out, src, index):
            return torch.scatter_add(out, 0, index, src)

        self._stage_and_e2e(kernel, out, src, index, expect=SCATTER_OP_SPEC)

    # ------------- Shape / index coverage for working scatters -------------
    # The scenarios above all share one fixed shape (M=128, N=256, P=3) from
    # _row_store/_full_index_store. These pin the same already-working ops
    # (index_put, scatter, scatter_add, index_copy, index_add) across stick
    # alignment, rank, index count, and index dtype so a shape-specific
    # regression surfaces here instead of only at the one baseline shape.
    def test_index_put_stick_aligned(self):
        """out[idx] = src, 1 stick per row (N=64), single-index edge (P=1)."""
        out, src, idx = self._row_store_shape(M=64, N=64, P=1)

        def kernel(out, src, idx):
            out[idx] = src
            return out

        self._stage_and_e2e(
            kernel, out, src, idx, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_index_put_non_stick_aligned(self):
        """out[idx] = src, non-stick-aligned row width (N=100)."""
        out, src, idx = self._row_store_shape(M=67, N=100, P=5)

        def kernel(out, src, idx):
            out[idx] = src
            return out

        self._stage_and_e2e(
            kernel, out, src, idx, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_index_put_multistick_row(self):
        """out[idx] = src, row spans multiple sticks (N=512, 8 sticks)."""
        out, src, idx = self._row_store_shape(M=128, N=512, P=8)

        def kernel(out, src, idx):
            out[idx] = src
            return out

        self._stage_and_e2e(
            kernel, out, src, idx, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_index_put_all_rows(self):
        """out[idx] = src where P == M -- every row of out is scattered into,
        the boundary opposite of the P=1 single-index edge."""
        out, src, idx = self._row_store_shape(M=16, N=64, P=16)

        def kernel(out, src, idx):
            out[idx] = src
            return out

        self._stage_and_e2e(
            kernel, out, src, idx, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_index_put_int64_index(self):
        """out[idx] = src with an int64 index (vs. the int32 baseline)."""
        out, src, idx = self._row_store_shape(dtype=torch.int64)

        def kernel(out, src, idx):
            out[idx] = src
            return out

        self._stage_and_e2e(
            kernel, out, src, idx, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_index_put_3d(self):
        """out[idx] = src on a 3-D out/src, indexing dim 0 -- the row-store
        pattern one rank up from the 2-D baseline."""
        A, B, C, P = 32, 8, 64, 4
        out_layout = canonical_device_layout((A, B, C), torch.float16)
        out = torch.zeros(A, B, C, dtype=torch.float16).to(device_layout=out_layout)
        src = torch.rand(P, B, C, dtype=torch.float16).to("spyre")
        idx = torch.randperm(A)[:P].to(torch.int32).to("spyre")
        self.name_dims(out, {"A": A, "B": B, "C": C})
        self.name_dims(src, {"P": P, "B": B, "C": C})
        self.name_dims(idx, {"P": P})

        def kernel(out, src, idx):
            out[idx] = src
            return out

        self._stage_and_e2e(
            kernel, out, src, idx, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_paged_kv_store(self):
        """Paged KV-cache store: cache.view(CS*BS, H, D)[slot_idxs] = keys.

        Mirrors test_paged_kv in the gather file (same cache shape, same
        slot-index scheme) but for the write side: writing new KV entries into
        a flattened paged cache at scattered slot positions, the store half of
        a paged-attention KV cache update.

        The cache row (H*D = 1024 elems = 16 sticks) is indirectly accessed
        along the leading (cache, block) dims, so those dims need an explicit
        STL with cache & block as the OUTERMOST device dims -- dim_order
        [2, 0, 1, 3] on [cache_size, block_size, H, D] gives device dims
        [cache_size, D//64, block_size, H, 64] (stick stays on D). Without
        this the scatter only writes the first stick of each row.
        """
        cache_size, block_size, H, D, Lk = 512, 64, 8, 128, 64
        total_slots = cache_size * block_size
        kv_cache_layout = SpyreTensorLayout(
            [cache_size, block_size, H, D],
            [block_size * H * D, H * D, D, 1],
            torch.float16,
            [2, 0, 1, 3],
        )
        cache = torch.zeros(cache_size, block_size, H, D, dtype=torch.float16).to(
            device_layout=kv_cache_layout
        )
        keys = torch.rand(Lk, H, D, dtype=torch.float16).to("spyre")
        slot_idxs = torch.randperm(total_slots)[:Lk].to(torch.int64).to("spyre")

        def kernel(cache, keys, slot_idxs):
            cache.view(total_slots, H, D)[slot_idxs] = keys
            return cache

        self._stage_and_e2e(
            kernel, cache, keys, slot_idxs, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_scatter_stick_aligned(self):
        """torch.scatter(out, 0, index, src), 1 stick per row (N=64)."""
        out, src, index = self._full_index_store_shape(
            M=64, N=64, P=1, dtype=torch.int64
        )

        def kernel(out, src, index):
            return torch.scatter(out, 0, index, src)

        self._stage_and_e2e(
            kernel, out, src, index, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_scatter_non_stick_aligned(self):
        """torch.scatter(out, 0, index, src), non-stick-aligned row (N=100)."""
        out, src, index = self._full_index_store_shape(
            M=67, N=100, P=5, dtype=torch.int64
        )

        def kernel(out, src, index):
            return torch.scatter(out, 0, index, src)

        self._stage_and_e2e(
            kernel, out, src, index, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_scatter_multistick_row(self):
        """torch.scatter(out, 0, index, src), row spans multiple sticks (N=512)."""
        out, src, index = self._full_index_store_shape(
            M=128, N=512, P=8, dtype=torch.int64
        )

        def kernel(out, src, index):
            return torch.scatter(out, 0, index, src)

        self._stage_and_e2e(
            kernel, out, src, index, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_scatter_all_rows(self):
        """torch.scatter(out, 0, index, src) where P == M."""
        out, src, index = self._full_index_store_shape(
            M=16, N=64, P=16, dtype=torch.int64
        )

        def kernel(out, src, index):
            return torch.scatter(out, 0, index, src)

        self._stage_and_e2e(
            kernel, out, src, index, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_scatter_add_stick_aligned(self):
        """y.scatter_add_(0, index, src), 1 stick per row (N=64)."""
        out, src, index = self._full_index_store_shape(M=64, N=64, P=1)

        def kernel(out, src, index):
            return out.scatter_add_(0, index, src)

        self._stage_and_e2e(
            kernel, out, src, index, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_scatter_add_non_stick_aligned(self):
        """y.scatter_add_(0, index, src), non-stick-aligned row (N=100)."""
        out, src, index = self._full_index_store_shape(M=67, N=100, P=5)

        def kernel(out, src, index):
            return out.scatter_add_(0, index, src)

        self._stage_and_e2e(
            kernel, out, src, index, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_scatter_add_small_table(self):
        """y.scatter_add_(0, index, src), small out table (M=8, N=64)."""
        out, src, index = self._full_index_store_shape(M=8, N=64, P=4)

        def kernel(out, src, index):
            return out.scatter_add_(0, index, src)

        self._stage_and_e2e(
            kernel, out, src, index, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_scatter_add_large_table(self):
        """y.scatter_add_(0, index, src), large out table (M=256, N=256)."""
        out, src, index = self._full_index_store_shape(M=256, N=256, P=32)

        def kernel(out, src, index):
            return out.scatter_add_(0, index, src)

        self._stage_and_e2e(
            kernel, out, src, index, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_index_copy_stick_aligned(self):
        """torch.index_copy(out, 0, idx, src), 1 stick per row (N=64)."""
        out, src, idx = self._row_store_shape(M=64, N=64, P=1, dtype=torch.int64)

        def kernel(out, src, idx):
            return torch.index_copy(out, 0, idx, src)

        self._stage_and_e2e(
            kernel, out, src, idx, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_index_copy_non_stick_aligned(self):
        """torch.index_copy(out, 0, idx, src), non-stick-aligned row (N=100)."""
        out, src, idx = self._row_store_shape(M=67, N=100, P=5, dtype=torch.int64)

        def kernel(out, src, idx):
            return torch.index_copy(out, 0, idx, src)

        self._stage_and_e2e(
            kernel, out, src, idx, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_index_copy_multistick_row(self):
        """torch.index_copy(out, 0, idx, src), row spans multiple sticks (N=512)."""
        out, src, idx = self._row_store_shape(M=128, N=512, P=8, dtype=torch.int64)

        def kernel(out, src, idx):
            return torch.index_copy(out, 0, idx, src)

        self._stage_and_e2e(
            kernel, out, src, idx, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_index_add_stick_aligned(self):
        """out.index_add_(0, idx, src), 1 stick per row (N=64)."""
        out, src, idx = self._row_store_shape(M=64, N=64, P=1)

        def kernel(out, src, idx):
            return out.index_add_(0, idx, src)

        self._stage_and_e2e(
            kernel, out, src, idx, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_index_add_non_stick_aligned(self):
        """out.index_add_(0, idx, src), non-stick-aligned row (N=100)."""
        out, src, idx = self._row_store_shape(M=67, N=100, P=5)

        def kernel(out, src, idx):
            return out.index_add_(0, idx, src)

        self._stage_and_e2e(
            kernel, out, src, idx, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_index_add_small_table(self):
        """out.index_add_(0, idx, src), small out table (M=8, N=64)."""
        out, src, idx = self._row_store_shape(M=8, N=64, P=4)

        def kernel(out, src, idx):
            return out.index_add_(0, idx, src)

        self._stage_and_e2e(
            kernel, out, src, idx, expect=SCATTER_OP_SPEC, expect_close=True
        )

    def test_index_add_large_table(self):
        """out.index_add_(0, idx, src), large out table (M=256, N=256)."""
        out, src, idx = self._row_store_shape(M=256, N=256, P=32)

        def kernel(out, src, idx):
            return out.index_add_(0, idx, src)

        self._stage_and_e2e(
            kernel, out, src, idx, expect=SCATTER_OP_SPEC, expect_close=True
        )

    # ------------- Not Detected As Indirect Access Scatter -------------
    def test_scatter_reduce_amax(self):
        """out.scatter_reduce_(0, index, src, "amax")"""
        out, src, index = self._full_index_store(dtype=torch.int64)

        def kernel(out, src, index):
            return out.scatter_reduce_(0, index, src, "amax")

        self._stage_and_e2e(kernel, out, src, index, expect=DIRECT_OP_SPEC)

    def test_scatter_reduce_amin(self):
        """out.scatter_reduce_(0, index, src, "amin")"""
        out, src, index = self._full_index_store(dtype=torch.int64)

        def kernel(out, src, index):
            return out.scatter_reduce_(0, index, src, "amin")

        self._stage_and_e2e(kernel, out, src, index, expect=DIRECT_OP_SPEC)

    def test_scatter_reduce_prod(self):
        """out.scatter_reduce_(0, index, src, "prod")"""
        out, src, index = self._full_index_store(dtype=torch.int64)

        def kernel(out, src, index):
            return out.scatter_reduce_(0, index, src, "prod")

        self._stage_and_e2e(kernel, out, src, index, expect=DIRECT_OP_SPEC)

    # -- Known crashes (separate from the indirect-store path) -------------
    def test_index_fill_crashes(self):
        """out.index_fill_(0, idx, 0.0) -- scalar fill -> rank-0 Constant codegen."""
        out = torch.rand(128, 256, dtype=torch.float16).to("spyre")
        idx = torch.randint(0, 128, (3,), dtype=torch.int32).to("spyre")
        self.name_dims(out, {"M": 128, "N": 256})
        self.name_dims(idx, {"P": 3})

        def kernel(out, idx):
            return out.index_fill_(0, idx, 0.0)

        self.check(kernel, out, idx, expect=CRASHED)

    def test_masked_scatter_element_mask_unsupported(self):
        """Element-level mask (stride(-1) != 0): the decomposition rejects it.

        A full [M, N] per-element mask is not broadcast along the last dim, so
        spyre_masked_scatter raises Unsupported -- an element scatter would index
        into a packed 1-D source, and a lane within a stick is not addressable on
        Spyre. The raise propagates out of torch.compile before any op spec is
        produced (CRASHED in this harness); there is no bundle to run e2e.
        """
        M, N = 64, 64
        out = torch.zeros(M, N, dtype=torch.float16).to("spyre")
        mask = torch.randint(0, 2, (M, N), dtype=torch.bool).to("spyre")
        src = torch.rand(M, N, dtype=torch.float16).to("spyre")
        self.name_dims(out, {"M": M, "N": N})

        def kernel(out, mask, src):
            return torch.masked_scatter(out, mask, src)

        self.check(kernel, out, mask, src, expect=CRASHED)

    # -- Supported masked_scatter: row-broadcast mask lowers to a gather -------
    def test_masked_scatter_row_broadcast(self):
        """torch.masked_scatter with a mask broadcast along the last dim.

        A row-broadcast mask (stride(-1) == 0 -- e.g. an attention mask [B, S]
        expanded to [B, S, C]) selects whole rows, so spyre_masked_scatter lowers
        it to a stick gather source_2d[row_idx] plus a where. That is an indirect
        *read*, hence a GATHER_OP_SPEC (not an indirect-output scatter).
        """
        ROWS, COLS, SRC_ROWS, N_TRUE = 855, 5120, 266, 266
        inp = torch.rand(1, ROWS, COLS, dtype=torch.float16).to("spyre")
        src = torch.rand(SRC_ROWS, COLS, dtype=torch.float16).to("spyre")
        # Row-broadcast mask: [1, ROWS] expanded to [1, ROWS, COLS] (stride(-1)==0).
        mask_1d = torch.zeros(1, ROWS, dtype=torch.bool)
        mask_1d[0, torch.randperm(ROWS)[:N_TRUE]] = True
        mask = mask_1d.unsqueeze(-1).to("spyre").expand(1, ROWS, COLS)
        self.name_dims(inp, {"B": 1, "ROWS": ROWS, "COLS": COLS})
        self.name_dims(src, {"SRC_ROWS": SRC_ROWS, "COLS": COLS})

        def kernel(inp, mask, src):
            return torch.masked_scatter(inp, mask, src)

        self._stage_and_e2e(
            kernel, inp, mask, src, expect=GATHER_OP_SPEC, expect_close=True
        )

    def _row_broadcast_operands(self, shape, src_rows, n_true):
        """Supported masked_scatter operands: `self` of `shape`, a mask broadcast
        along the last dim (stride(-1) == 0) with `n_true` selected rows, and a
        2-D `src[src_rows, shape[-1]]`. Returns (inp, mask, src) on "spyre"."""
        *lead, cols = shape
        rows = 1
        for d in lead:
            rows *= d
        inp = torch.rand(*shape, dtype=torch.float16).to("spyre")
        src = torch.rand(src_rows, cols, dtype=torch.float16).to("spyre")
        flat = torch.zeros(rows, dtype=torch.bool)
        flat[torch.randperm(rows)[:n_true]] = True
        mask = flat.reshape(*lead, 1).to("spyre").expand(*shape)
        # Share the column dim name "C" across self and src (house convention:
        # the two operands' common last dim is one named dim, cf. _row_store).
        lead_names = [f"L{i}" for i in range(len(lead))]
        self.name_dims(inp, dict(zip(lead_names + ["C"], list(lead) + [cols])))
        self.name_dims(src, {"SRC": src_rows, "C": cols})
        return inp, mask, src

    def test_masked_scatter_2d_row_broadcast(self):
        """2-D self [M, N] with a row-broadcast mask -- the smallest supported
        shape. Capture-only: assert it reaches a gather op spec (and a valid
        indirect-access SDSC bundle), no device run needed."""
        inp, mask, src = self._row_broadcast_operands(
            (128, 256), src_rows=40, n_true=40
        )

        def kernel(inp, mask, src):
            return torch.masked_scatter(inp, mask, src)

        self.check(kernel, inp, mask, src, expect=GATHER_OP_SPEC)

    def test_masked_scatter_batched_row_broadcast(self):
        """Batched self [B, S, C] (rows = B*S) with a row-broadcast mask. Pins
        that a leading batch dim still collapses to whole-row selection and
        reaches a gather op spec."""
        inp, mask, src = self._row_broadcast_operands(
            (2, 64, 128), src_rows=40, n_true=40
        )

        def kernel(inp, mask, src):
            return torch.masked_scatter(inp, mask, src)

        self.check(kernel, inp, mask, src, expect=GATHER_OP_SPEC)

    def test_masked_scatter_all_rows_selected(self):
        """All-True mask: every row is selected, so src must supply exactly `rows`
        rows and the result is `src` row-for-row. Exercises the max-selection
        boundary end-to-end."""
        M, N = 8, 128
        inp = torch.rand(M, N, dtype=torch.float16).to("spyre")
        src = torch.rand(M, N, dtype=torch.float16).to("spyre")
        mask = torch.ones(M, 1, dtype=torch.bool).to("spyre").expand(M, N)
        self.name_dims(inp, {"M": M, "N": N})
        self.name_dims(src, {"M": M, "N": N})

        def kernel(inp, mask, src):
            return torch.masked_scatter(inp, mask, src)

        self._stage_and_e2e(
            kernel, inp, mask, src, expect=GATHER_OP_SPEC, expect_close=True
        )

    def test_masked_scatter_no_rows_selected(self):
        """All-False mask: nothing is selected, so the result must equal `self`
        unchanged. The gather still lowers (the mask is a runtime input, not a
        compile-time constant), so this stays a GATHER_OP_SPEC."""
        M, N = 8, 128
        inp = torch.rand(M, N, dtype=torch.float16).to("spyre")
        src = torch.rand(4, N, dtype=torch.float16).to("spyre")
        mask = torch.zeros(M, 1, dtype=torch.bool).to("spyre").expand(M, N)
        self.name_dims(inp, {"M": M, "N": N})
        self.name_dims(src, {"SRC": 4, "N": N})

        def kernel(inp, mask, src):
            return torch.masked_scatter(inp, mask, src)

        self._stage_and_e2e(
            kernel, inp, mask, src, expect=GATHER_OP_SPEC, expect_close=True
        )

    def test_masked_scatter_degenerate_last_dim_unsupported(self):
        """Degenerate last dim (cols == 1): a single-column row is not a real
        shared row for the block-per-row equivalence, so the decomposition
        rejects it with Unsupported (CRASHED here)."""
        M = 128
        inp = torch.rand(M, 1, dtype=torch.float16).to("spyre")
        src = torch.rand(4, 1, dtype=torch.float16).to("spyre")
        mask = torch.zeros(M, 1, dtype=torch.bool)
        mask[torch.randperm(M)[:4]] = True
        mask = mask.to("spyre")
        self.name_dims(inp, {"M": M, "ONE": 1})

        def kernel(inp, mask, src):
            return torch.masked_scatter(inp, mask, src)

        self.check(kernel, inp, mask, src, expect=CRASHED)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
