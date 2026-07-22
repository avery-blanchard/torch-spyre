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

"""Tests for mutation-layout restickify gap in propagate_mutation_layouts.

These tests exercise scenarios where a mutation target is not directly a
graph input (so _target_device_layout returns None) and must go through
propagate_mutation_layouts. The scenarios combine slicing + transposition
to create stick-layout conflicts that should trigger restickify insertion.

On main (pre-fix), these may compile but produce wrong results at runtime
on actual Spyre hardware due to the restickify gap.
"""

import unittest

import torch

from torch_spyre.constants import DEVICE_NAME


class TestMutationLayoutRestickifyGap(unittest.TestCase):
    """Mutation layout scenarios that expose the propagate_mutation_layouts gap."""

    def _compare(self, fn, *args):
        """Run fn on CPU (eager) and Spyre (compiled), compare outputs."""
        # CPU eager
        cpu_args = [a.clone().cpu() for a in args]
        cpu_result = fn(*cpu_args)

        # Spyre compiled
        spyre_args = [a.clone().to(DEVICE_NAME) for a in args]
        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False)
        spyre_result = compiled(*spyre_args)
        spyre_result = (
            spyre_result.cpu()
            if isinstance(spyre_result, torch.Tensor)
            else [t.cpu() for t in spyre_result]
        )

        # Compare
        if isinstance(cpu_result, tuple):
            for cpu_r, spyre_r in zip(cpu_result, spyre_result):
                self.assertTrue(
                    torch.allclose(cpu_r, spyre_r, atol=1e-2, rtol=1e-2),
                    f"Mismatch: CPU {cpu_r.shape} vs Spyre {spyre_r.shape}",
                )
        else:
            self.assertTrue(
                torch.allclose(cpu_result, spyre_result, atol=1e-2, rtol=1e-2),
                f"Mismatch: CPU {cpu_result.shape} vs Spyre {spyre_result.shape}",
            )

    def test_slice_add_with_transposed_rhs(self):
        """Mutate a graph-input slice via add_ with a transposed RHS."""

        def fn(buf, x):
            buf[:192].add_(x)
            return buf

        buf = torch.randn(256, 192, dtype=torch.float16)
        x = torch.randn(192, 192, dtype=torch.float16)
        self._compare(fn, buf, x)

    def test_slice_mul_with_transposed_rhs(self):
        """Mutate a graph-input slice via mul_ with a transposed RHS."""

        def fn(buf, x):
            buf[:192].mul_(x)
            return buf

        buf = torch.randn(256, 192, dtype=torch.float16)
        x = torch.randn(192, 192, dtype=torch.float16)
        self._compare(fn, buf, x)

    def test_slice_add_then_read_original(self):
        """Mutate a slice, then read the full buffer (tests copy-back correctness)."""

        def fn(buf, x):
            buf[:192].add_(x)
            return buf.sum()

        buf = torch.randn(256, 192, dtype=torch.float16)
        x = torch.randn(192, 192, dtype=torch.float16)
        self._compare(fn, buf, x)

    def test_slice_add_then_matmul(self):
        """Mutate a slice, then use mutated buffer in matmul."""

        def fn(buf, x, w):
            buf[:192].add_(x)
            return buf @ w

        buf = torch.randn(256, 192, dtype=torch.float16)
        x = torch.randn(192, 192, dtype=torch.float16)
        w = torch.randn(192, 128, dtype=torch.float16)
        self._compare(fn, buf, x, w)

    def test_column_slice_add_transposed(self):
        """Mutate a column slice with transposed RHS."""

        def fn(buf, x):
            buf[:, :192].add_(x)
            return buf

        buf = torch.randn(192, 256, dtype=torch.float16)
        x = torch.randn(192, 192, dtype=torch.float16)
        self._compare(fn, buf, x)

    def test_strided_slice_add_transposed(self):
        """Mutate a strided slice with transposed RHS."""

        def fn(buf, x):
            buf[::4].add_(x)
            return buf

        buf = torch.randn(256, 192, dtype=torch.float16)
        x = torch.randn(64, 192, dtype=torch.float16)
        self._compare(fn, buf, x)

    def test_multiple_mutations_same_buffer(self):
        """Multiple mutations on same buffer (tests layout consistency)."""

        def fn(buf, x, y):
            buf[:128].add_(x)
            buf[128:].mul_(y)
            return buf

        buf = torch.randn(256, 192, dtype=torch.float16)
        x = torch.randn(128, 192, dtype=torch.float16)
        y = torch.randn(128, 192, dtype=torch.float16)
        self._compare(fn, buf, x, y)

    def test_mutation_with_two_downstream_consumers(self):
        """Mutate, then consume mutated buffer in two separate ops."""

        def fn(buf, x, w):
            buf[:192].add_(x)
            out1 = buf @ w
            out2 = buf.sum()
            return out1, out2

        buf = torch.randn(256, 192, dtype=torch.float16)
        x = torch.randn(192, 192, dtype=torch.float16)
        w = torch.randn(192, 128, dtype=torch.float16)
        self._compare(fn, buf, x, w)

    def test_slice_of_slice_mutation(self):
        """Mutate a slice-of-slice (nested slicing)."""

        def fn(buf, x):
            buf[64:192, :128].add_(x)
            return buf

        buf = torch.randn(256, 256, dtype=torch.float16)
        x = torch.randn(128, 128, dtype=torch.float16)
        self._compare(fn, buf, x)


if __name__ == "__main__":
    unittest.main()
