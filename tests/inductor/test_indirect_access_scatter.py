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

The two forms that crash during compilation -- index_fill (rank-0 scalar
Constant codegen) and masked_scatter (mask-based CPU fallback) -- stay
capture-only via check(expect=CRASHED); there is no bundle to run end-to-end.

All scatter scenarios run with SENCORES=1.

Status (validated on hardware build): index-tensor scatters reach a real op
spec with IndirectAccess on the output (SCATTER_OP_SPEC); the deeptools backend
diverges from / aborts on the bundle, surfaced here as xfail.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from indirect_access_common import (  # noqa: E402
    CRASHED,
    SCATTER_OP_SPEC,
    DIRECT_OP_SPEC,
    IndirectAccessTestCase,
    arg_has_indirect_access,
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

    # -- Working index-tensor scatters: op spec with output IndirectAccess --
    def test_index_put(self):
        """out[idx] = src"""
        out, src, idx = self._row_store()

        def kernel(out, src, idx):
            out[idx] = src
            return out

        self.check(kernel, out, src, idx, expect=SCATTER_OP_SPEC)

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

        self.check(kernel, out, src, idx, expect=SCATTER_OP_SPEC)

    def test_index_add(self):
        """out.index_add_(0, idx, src)"""
        out, src, idx = self._row_store()

        def kernel(out, src, idx):
            return out.index_add_(0, idx, src)

        self.check(kernel, out, src, idx, expect=SCATTER_OP_SPEC)

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

        self.check(kernel, out, src, idx, expect=SCATTER_OP_SPEC)

    def test_scatter_add_functional(self):
        """torch.scatter_add(out, 0, index, src) -- functional accumulating scatter."""
        out, src, index = self._full_index_store()

        def kernel(out, src, index):
            return torch.scatter_add(out, 0, index, src)

        self._stage_and_e2e(kernel, out, src, index, expect=SCATTER_OP_SPEC)

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

    def test_masked_scatter_crashes(self):
        """torch.masked_scatter(out, mask, src) -- uses mask-based CPU fallback path."""
        M, N = 64, 64
        out = torch.zeros(M, N, dtype=torch.float16).to("spyre")
        mask = torch.randint(0, 2, (M, N), dtype=torch.bool).to("spyre")
        src = torch.rand(M, N, dtype=torch.float16).to("spyre")
        self.name_dims(out, {"M": M, "N": N})

        def kernel(out, mask, src):
            return torch.masked_scatter(out, mask, src)

        self.check(kernel, out, mask, src, expect=CRASHED)


@config.patch({"sencores": 1})
class TestScatterLayoutConstraint(IndirectAccessTestCase):
    """Test that scatter operations enforce indirect dimension at device dim 0."""

    def assert_indirect_at_device_dim_0(self, op_specs):
        """Assert that scatter output has IndirectAccess at device coordinate position 0."""
        from torch_spyre._inductor.op_spec import IndirectAccess

        found_scatter_output = False
        for spec in op_specs:
            for arg in spec.args:
                if not arg.is_input and arg_has_indirect_access(arg):
                    found_scatter_output = True
                    first_coord = arg.device_coordinates[0]
                    self.assertTrue(
                        isinstance(first_coord, IndirectAccess),
                        f"Scatter output {arg.name}: expected IndirectAccess at device dim 0, "
                        f"got {first_coord} in coordinates {arg.device_coordinates}",
                    )
        self.assertTrue(
            found_scatter_output,
            "no op spec had an IndirectAccess output arg to check",
        )

    def test_scatter_2d_1d_index(self):
        """2D output [1024, 3], 1D index [3]: indirect dim should be outermost."""
        out = torch.zeros(1024, 3, dtype=torch.float16).to("spyre")
        src = torch.rand(3, 3, dtype=torch.float16).to("spyre")
        idx = torch.randint(0, 1024, (3,), dtype=torch.int32).to("spyre")
        self.name_dims(out, {"M": 1024, "N": 3})
        self.name_dims(src, {"P": 3, "N": 3})
        self.name_dims(idx, {"P": 3})

        def kernel(out, src, idx):
            out[idx] = src
            return out

        r = self.check(kernel, out, src, idx, expect=SCATTER_OP_SPEC)
        self.assert_indirect_at_device_dim_0(r.op_specs)

    def test_scatter_2d_2d_index(self):
        """2D output [1024, 64], 2D index [32, 64]: indirect dim at dim 0."""
        out = torch.zeros(1024, 64, dtype=torch.float16).to("spyre")
        src = torch.rand(32, 64, dtype=torch.float16).to("spyre")
        idx = torch.randint(0, 1024, (32, 64), dtype=torch.int64).to("spyre")
        self.name_dims(out, {"M": 1024, "N": 64})
        self.name_dims(src, {"P": 32, "N": 64})
        self.name_dims(idx, {"P": 32, "N": 64})

        def kernel(out, src, idx):
            return torch.scatter(out, 0, idx, src)

        r = self.check(kernel, out, src, idx, expect=SCATTER_OP_SPEC)
        self.assert_indirect_at_device_dim_0(r.op_specs)

    def test_scatter_3d_output(self):
        """3D output [512, 128, 8], 1D index [16]: indirect dim outermost."""
        out = torch.zeros(512, 128, 8, dtype=torch.float16).to("spyre")
        src = torch.rand(16, 128, 8, dtype=torch.float16).to("spyre")
        idx = torch.randint(0, 512, (16,), dtype=torch.int32).to("spyre")
        self.name_dims(out, {"M": 512, "N": 128, "K": 8})
        self.name_dims(src, {"P": 16, "N": 128, "K": 8})
        self.name_dims(idx, {"P": 16})

        def kernel(out, src, idx):
            out[idx] = src
            return out

        r = self.check(kernel, out, src, idx, expect=SCATTER_OP_SPEC)
        self.assert_indirect_at_device_dim_0(r.op_specs)

    def test_scatter_4d_output(self):
        """4D output [256, 256, 16, 16], 2D index [32, 16]: indirect at dim 0."""
        out = torch.zeros(256, 256, 16, 16, dtype=torch.float16).to("spyre")
        src = torch.rand(32, 256, 16, 16, dtype=torch.float16).to("spyre")
        idx = torch.randint(0, 256, (32, 256, 16, 16), dtype=torch.int64).to("spyre")
        self.name_dims(out, {"A": 256, "B": 256, "C": 16, "D": 16})
        self.name_dims(src, {"P": 32, "B": 256, "C": 16, "D": 16})
        self.name_dims(idx, {"P": 32, "B": 256, "C": 16, "D": 16})

        def kernel(out, src, idx):
            return torch.scatter(out, 0, idx, src)

        r = self.check(kernel, out, src, idx, expect=SCATTER_OP_SPEC)
        self.assert_indirect_at_device_dim_0(r.op_specs)

    def test_scatter_3d_2d_index(self):
        """3D output [1024, 64, 16], 2D index [128, 64]: indirect dim outermost."""
        out = torch.zeros(1024, 64, 16, dtype=torch.float16).to("spyre")
        src = torch.rand(128, 64, 16, dtype=torch.float16).to("spyre")
        idx = torch.randint(0, 1024, (128, 64, 16), dtype=torch.int64).to("spyre")
        self.name_dims(out, {"M": 1024, "N": 64, "K": 16})
        self.name_dims(src, {"P": 128, "N": 64, "K": 16})
        self.name_dims(idx, {"P": 128, "N": 64, "K": 16})

        def kernel(out, src, idx):
            return out.scatter_(0, idx, src)

        r = self.check(kernel, out, src, idx, expect=SCATTER_OP_SPEC)
        self.assert_indirect_at_device_dim_0(r.op_specs)

    def test_scatter_add_2d(self):
        """scatter_add with 2D output and 1D index: indirect dim at device 0."""
        out = torch.zeros(512, 64, dtype=torch.float16).to("spyre")
        src = torch.rand(32, 64, dtype=torch.float16).to("spyre")
        idx = torch.randint(0, 512, (32, 64), dtype=torch.int64).to("spyre")
        self.name_dims(out, {"M": 512, "N": 64})
        self.name_dims(src, {"P": 32, "N": 64})
        self.name_dims(idx, {"P": 32, "N": 64})

        def kernel(out, src, idx):
            return out.scatter_add_(0, idx, src)

        r = self.check(kernel, out, src, idx, expect=SCATTER_OP_SPEC)
        self.assert_indirect_at_device_dim_0(r.op_specs)

    def test_scatter_transposed_output(self):
        """Scatter with transposed (non-contiguous) output view.
        [1024, 64] transposed to [64, 1024], then scattered on dim 0 via
        index [32]. Tests that stride calculation in _indirect_write_host_dim
        and the pinned layout construction handle non-standard strides.
        """
        out_base = torch.zeros(64, 1024, dtype=torch.float16).to("spyre")
        out = out_base.t()  # Transpose: now [1024, 64], strides [1, 1024]
        src = torch.rand(32, 64, dtype=torch.float16).to("spyre")
        idx = torch.randint(0, 1024, (32,), dtype=torch.int32).to("spyre")
        self.name_dims(out, {"M": 1024, "N": 64})
        self.name_dims(src, {"P": 32, "N": 64})
        self.name_dims(idx, {"P": 32})

        def kernel(out, src, idx):
            out[idx] = src
            return out

        r = self.check(kernel, out, src, idx, expect=SCATTER_OP_SPEC)
        self.assert_indirect_at_device_dim_0(r.op_specs)

    def test_scatter_strided_view(self):
        """Scatter with a strided view (sub-sampling every other element).
        3D output [512, 16, 64] → view with step 2 on dim 1 → [512, 8, 64],
        then scattered on dim 0. Tests that non-unit strides in non-scattered
        dims are correctly captured and singleton handling works with them.
        """
        out_base = torch.zeros(512, 16, 64, dtype=torch.float16).to("spyre")
        out = out_base[:, ::2, :]  # Step 2 on dim 1: [512, 8, 64]
        src = torch.rand(32, 8, 64, dtype=torch.float16).to("spyre")
        idx = torch.randint(0, 512, (32,), dtype=torch.int32).to("spyre")
        self.name_dims(out, {"M": 512, "N": 8, "K": 64})
        self.name_dims(src, {"P": 32, "N": 8, "K": 64})
        self.name_dims(idx, {"P": 32})

        def kernel(out, src, idx):
            out[idx] = src
            return out

        r = self.check(kernel, out, src, idx, expect=SCATTER_OP_SPEC)
        self.assert_indirect_at_device_dim_0(r.op_specs)

    def test_scatter_3d_singleton_trailing_dim(self):
        """3D output [512, 4, 1], 1D index [16]: a non-pinned, non-stick dim has
        size 1. This must produce a genuine singleton (stride_map -1) device
        dim rather than one carrying a stray host stride -- regression test
        for the manual device_size/stride_map construction used once the
        indirect-index dim is pinned to device dim 0.
        """
        out = torch.zeros(512, 4, 1, dtype=torch.float16).to("spyre")
        src = torch.rand(16, 4, 1, dtype=torch.float16).to("spyre")
        idx = torch.randint(0, 512, (16,), dtype=torch.int32).to("spyre")
        self.name_dims(out, {"M": 512, "N": 4, "K": 1})
        self.name_dims(src, {"P": 16, "N": 4, "K": 1})
        self.name_dims(idx, {"P": 16})

        def kernel(out, src, idx):
            out[idx] = src
            return out

        r = self.check(kernel, out, src, idx, expect=SCATTER_OP_SPEC)
        self.assert_indirect_at_device_dim_0(r.op_specs)

    def test_scatter_4d_singleton_middle_dim(self):
        """4D output [256, 1, 16, 16], 1D index [32]: a non-pinned, non-stick,
        non-trailing dim has size 1. Exercises the same singleton-stride case
        as test_scatter_3d_singleton_trailing_dim but for a middle dim, to
        confirm the fix isn't position-dependent.
        """
        out = torch.zeros(256, 1, 16, 16, dtype=torch.float16).to("spyre")
        src = torch.rand(32, 1, 16, 16, dtype=torch.float16).to("spyre")
        idx = torch.randint(0, 256, (32,), dtype=torch.int32).to("spyre")
        self.name_dims(out, {"A": 256, "B": 1, "C": 16, "D": 16})
        self.name_dims(src, {"P": 32, "B": 1, "C": 16, "D": 16})
        self.name_dims(idx, {"P": 32})

        def kernel(out, src, idx):
            out[idx] = src
            return out

        r = self.check(kernel, out, src, idx, expect=SCATTER_OP_SPEC)
        self.assert_indirect_at_device_dim_0(r.op_specs)

    def test_scatter_dim1_2d_output(self):
        """Scatter on dim 1 (not dim 0): 2D output [128, 256], 1D index [64].
        Tests that the layout constraint applies regardless of scatter dimension.
        """
        out = torch.zeros(128, 256, dtype=torch.float16).to("spyre")
        src = torch.rand(128, 64, dtype=torch.float16).to("spyre")
        idx = torch.randint(0, 256, (64,), dtype=torch.int32).to("spyre")
        self.name_dims(out, {"M": 128, "N": 256})
        self.name_dims(src, {"M": 128, "P": 64})
        self.name_dims(idx, {"P": 64})

        def kernel(out, src, idx):
            return torch.scatter(out, 1, idx, src)

        r = self.check(kernel, out, src, idx, expect=SCATTER_OP_SPEC)
        self.assert_indirect_at_device_dim_0(r.op_specs)

    def test_scatter_dim1_3d_output(self):
        """Scatter on dim 1 with 3D output [64, 512, 8], 1D index [32].
        Verifies layout constraint on intermediate dimension.
        """
        out = torch.zeros(64, 512, 8, dtype=torch.float16).to("spyre")
        src = torch.rand(64, 32, 8, dtype=torch.float16).to("spyre")
        idx = torch.randint(0, 512, (32,), dtype=torch.int32).to("spyre")
        self.name_dims(out, {"M": 64, "N": 512, "K": 8})
        self.name_dims(src, {"M": 64, "P": 32, "K": 8})
        self.name_dims(idx, {"P": 32})

        def kernel(out, src, idx):
            return torch.scatter(out, 1, idx, src)

        r = self.check(kernel, out, src, idx, expect=SCATTER_OP_SPEC)
        self.assert_indirect_at_device_dim_0(r.op_specs)

    def test_scatter_dim2_3d_output(self):
        """Scatter on dim 2 (innermost non-stick dim): 3D output [64, 128, 256].
        Tests layout constraint on the deepest scatter dimension.
        """
        out = torch.zeros(64, 128, 256, dtype=torch.float16).to("spyre")
        src = torch.rand(64, 128, 48, dtype=torch.float16).to("spyre")
        idx = torch.randint(0, 256, (48,), dtype=torch.int32).to("spyre")
        self.name_dims(out, {"M": 64, "N": 128, "K": 256})
        self.name_dims(src, {"M": 64, "N": 128, "P": 48})
        self.name_dims(idx, {"P": 48})

        def kernel(out, src, idx):
            return torch.scatter(out, 2, idx, src)

        r = self.check(kernel, out, src, idx, expect=SCATTER_OP_SPEC)
        self.assert_indirect_at_device_dim_0(r.op_specs)

    def test_scatter_dim2_4d_output(self):
        """Scatter on dim 2 with 4D output [32, 64, 128, 16].
        Exercises layout constraint with multiple trailing dimensions.
        """
        out = torch.zeros(32, 64, 128, 16, dtype=torch.float16).to("spyre")
        src = torch.rand(32, 64, 32, 16, dtype=torch.float16).to("spyre")
        idx = torch.randint(0, 128, (32,), dtype=torch.int32).to("spyre")
        self.name_dims(out, {"A": 32, "B": 64, "C": 128, "D": 16})
        self.name_dims(src, {"A": 32, "B": 64, "P": 32, "D": 16})
        self.name_dims(idx, {"P": 32})

        def kernel(out, src, idx):
            return torch.scatter(out, 2, idx, src)

        r = self.check(kernel, out, src, idx, expect=SCATTER_OP_SPEC)
        self.assert_indirect_at_device_dim_0(r.op_specs)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
