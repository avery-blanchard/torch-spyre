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

"""Compute and validate device-index terms for tensor layouts.

This pass runs during layout propagation. For each tensor that gets assigned
a SpyreTensorLayout, it computes the device-index terms from the layout's
stride_map and the tensor's host index expression, validating structural
invariants to catch indexing bugs early.
"""

from __future__ import annotations

import sympy

from .device_index import (
    DeviceIndexError,
    compute_device_index_terms,
    validate_device_index_terms,
)
from .errors import Unsupported
from .views import Term


def compute_indices_for_layout(
    stride_map: list[int],
    device_size: list[int],
    var_ranges: dict[sympy.Symbol, sympy.Expr],
    host_index: sympy.Expr,
    num_elements_per_stick: int,
    tensor_name: str = "",
) -> tuple[dict[sympy.Symbol, list[Term]], sympy.Symbol | None]:
    """Compute device-index terms for a tensor layout.

    Args:
        stride_map: Stride per device dimension
        device_size: Device extent per dimension
        var_ranges: Loop variable ranges (from dep.ranges)
        host_index: Host index expression
        num_elements_per_stick: Element count per stick for this tensor's dtype
        tensor_name: Name for error messages

    Returns:
        (grouped, stick_var): Device-index terms grouped by loop variable.

    Raises:
        Unsupported: If device-index computation or validation fails.
    """
    try:
        grouped, stick_var = compute_device_index_terms(
            var_ranges, device_size, stride_map, host_index
        )
        validate_device_index_terms(grouped, stick_var, num_elements_per_stick)
        return (grouped, stick_var)
    except DeviceIndexError as e:
        raise Unsupported(
            f"device-index validation failed for {tensor_name}: {e}"
        ) from e
