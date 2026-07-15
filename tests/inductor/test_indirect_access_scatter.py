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

        for spec in op_specs:
            # Find the output arg (is_input=False)
            for arg in spec.args:
                if (
                    not arg.is_input
                    and isinstance(spec.op, str)
                    and "scatter" in spec.op.lower()
                ):
                    # Check that device_coordinates[0] contains an IndirectAccess
                    if arg.device_coordinates:
                        first_coord = arg.device_coordinates[0]
                        self.assertTrue(
                            isinstance(first_coord, IndirectAccess)
                            or (
                                hasattr(first_coord, "has")
                                and first_coord.has(IndirectAccess)
                            ),
                            f"Scatter output {arg.name}: expected IndirectAccess at device dim 0, "
                            f"got {first_coord} in coordinates {arg.device_coordinates}",
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


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
