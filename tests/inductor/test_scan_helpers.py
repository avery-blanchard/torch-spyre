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


"""Unit tests for the scan helpers in torch_spyre._inductor.

``scan_axis`` and ``host_ordered_scan_space`` are pure functions of the IR
node and index expressions, so they are exercised here without hardware.
"""

from types import SimpleNamespace

import pytest
import sympy
import torch

import torch_spyre  # noqa: F401  (registers spyre::non_zero_pos_to_idx)
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.pass_utils import host_ordered_scan_space, scan_axis

_TARGET = torch.ops.spyre.non_zero_pos_to_idx.default


def _scan_node(*origin_calls, size=(4, 64, 128)):
    return SimpleNamespace(size=list(size), origins=list(origin_calls))


def _call(*args, **kwargs):
    return SimpleNamespace(target=_TARGET, args=args, kwargs=kwargs)


def test_scan_axis_positional_dim():
    assert scan_axis(_scan_node(_call("x", 1))) == 1


def test_scan_axis_normalizes_negative_dim():
    assert scan_axis(_scan_node(_call("x", -1))) == 2


def test_scan_axis_keyword_dim():
    assert scan_axis(_scan_node(_call("x", dim=0))) == 0


def test_scan_axis_ignores_other_origins():
    other = SimpleNamespace(
        target=torch.ops.aten.add.Tensor, args=("a", "b"), kwargs={}
    )
    assert scan_axis(_scan_node(other, _call("x", 2))) == 2


def test_scan_axis_rejects_missing_call():
    other = SimpleNamespace(
        target=torch.ops.aten.add.Tensor, args=("a", "b"), kwargs={}
    )
    with pytest.raises(Unsupported, match="0 non_zero_pos_to_idx origin calls"):
        scan_axis(_scan_node(other))


def test_scan_axis_rejects_non_literal_dim():
    with pytest.raises(Unsupported, match="not a literal int"):
        scan_axis(_scan_node(_call("x", sympy.Symbol("s"))))


class _Write:
    """Stand-in for a MemoryDep: only the index is read."""

    def __init__(self, index):
        self.index = index


def test_host_ordered_scan_space_follows_strides():
    d0, d1, d2 = sympy.symbols("d0 d1 d2")
    # Host layout (4, 128, 64) with d1 the 128-wide dim: strides 8192, 64, 1.
    write = _Write(8192 * d0 + 64 * d1 + d2)
    space = {d0: 4, d1: 128, d2: 64}  # the dep's own (non-host) order
    assert list(host_ordered_scan_space(space, write)) == [d0, d1, d2]


def test_host_ordered_scan_space_moves_scan_dim_to_host_position():
    d0, d1, d2 = sympy.symbols("d0 d1 d2")
    # Write dep lists pointwise vars first, so host dim 1 (d2) comes last here.
    write = _Write(8192 * d0 + d1 + 128 * d2)
    space = {d0: 4, d1: 128, d2: 64}
    assert list(host_ordered_scan_space(space, write)) == [d0, d2, d1]
