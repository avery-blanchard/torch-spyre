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

import torch_spyre  # noqa: F401  (registers spyre::non_zero_pos_to_idx)
from torch_spyre._inductor.errors import Unsupported
from torch._inductor.ir import Scan
from torch_spyre._inductor.pass_utils import (
    host_ordered_scan_space,
    scan_axis,
    scan_var as op_scan_var,
)


def _scan_op(pointwise, scan):
    data = Scan.__new__(Scan)
    object.__setattr__(data, "ranges", list(pointwise))
    object.__setattr__(data, "scan_ranges", list(scan))
    object.__setattr__(data, "size", list(pointwise) + list(scan))
    return SimpleNamespace(data=data, get_name=lambda: "buf0")


def _body(var, extent, write_index):
    """Stand-in for a LoopBody: its declared scan var and write index."""
    return SimpleNamespace(
        reduce_vars=[var],
        var_ranges={var: extent},
        get_write_expr=lambda name: write_index,
    )


def _rw(ranges, index):
    return SimpleNamespace(writes=[SimpleNamespace(ranges=ranges, index=index)])


def test_scan_var_matches_declared_scan_var_by_extent_and_stride():
    q0, q1 = sympy.symbols("q0 q1")
    c0, c1, c2 = sympy.symbols("c0 c1 c2")
    # Body: scan var q1 (extent 64, stride 128); deps rename it, any position.
    body = _body(q1, 64, 8192 * q0 + 128 * q1)
    rw = _rw({c0: 64, c1: 4}, 128 * c0 + 8192 * c1)
    assert op_scan_var(_scan_op([4], [64]), rw, body) == c0


def test_scan_var_rejects_merged_scan_dim():
    q0, c0 = sympy.symbols("q0 c0")
    body = _body(q0, 16, q0)
    rw = _rw({c0: 128}, c0)  # scan merged with an 8-wide neighbour
    with pytest.raises(Unsupported, match="matches \\[\\]"):
        op_scan_var(_scan_op([8], [16]), rw, body)


def test_scan_var_rejects_ambiguous_match():
    q0, c0, c1 = sympy.symbols("q0 c0 c1")
    body = _body(q0, 4, q0)
    rw = _rw({c0: 4, c1: 4}, c0 + c1)
    with pytest.raises(Unsupported, match="matches"):
        op_scan_var(_scan_op([4], [4]), rw, body)


def test_scan_var_rejects_non_scan():
    with pytest.raises(Unsupported, match="not a scan"):
        op_scan_var(SimpleNamespace(data=object(), get_name=lambda: "b"))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_scan_axis_from_reindex(axis):
    data = _scan_op([4, 128], [64]).data

    def reindex(index, scan_index):
        return [*index[:axis], *scan_index, *index[axis:]]

    object.__setattr__(data, "reindex", reindex)
    assert scan_axis(data) == axis


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
