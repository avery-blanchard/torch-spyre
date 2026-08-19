# Copyright 2026 The Torch-Spyre Authors.
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

import math

import pytest
import sympy

from torch_spyre._inductor.core_mapping import core_to_slice_mapping


def _coordinates(splits, num_cores, **kwargs):
    dims = sympy.symbols(f"dim_0:{len(splits)}")
    mapping = core_to_slice_mapping(dims, splits, num_cores, **kwargs)
    core_id = sympy.Symbol("core_id")
    return [
        tuple(int(mapping[dim].subs(core_id, core)) for dim in dims)
        for core in range(num_cores)
    ]


def test_default_mapping_preserves_existing_core_order():
    one_grid = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
    assert _coordinates((2, 3), 12) == one_grid * 2


@pytest.mark.parametrize("contiguous_dim", [0, 1, 2])
def test_selected_dim_varies_first(contiguous_dim):
    splits = (2, 3, 4)
    coordinates = _coordinates(
        splits,
        math.prod(splits),
        contiguous_dim=contiguous_dim,
    )
    assert [
        coordinate[contiguous_dim]
        for coordinate in coordinates[: splits[contiguous_dim]]
    ] == list(range(splits[contiguous_dim]))
    assert all(
        coordinate[dim] == 0
        for coordinate in coordinates[: splits[contiguous_dim]]
        for dim in range(len(splits))
        if dim != contiguous_dim
    )
