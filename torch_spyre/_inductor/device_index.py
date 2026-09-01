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

"""Device-index term computation and validation.

Computes how each loop variable contributes to the device address via a
stride_map, producing an ordered list of Terms grouped by traversal variable.
This replaces device_coordinates with an explicit term-structure representation
keyed by semantics (which loop var) rather than tensor shape (which device dim).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import sympy

from .views import Term


@dataclasses.dataclass(frozen=True)
class DeviceIndexError(Exception):
    """Raised when device-index term structure violates an invariant."""

    message: str
    var: sympy.Symbol | None = None
    details: str | None = None

    def __str__(self) -> str:
        s = self.message
        if self.var is not None:
            s += f" (var={self.var!r})"
        if self.details is not None:
            s += f": {self.details}"
        return s


def compute_device_index_terms(
    var_ranges: dict[sympy.Symbol, sympy.Expr],
    device_size: Sequence[int],
    stride_map: Sequence[int],
    index: sympy.Expr,
) -> tuple[dict[sympy.Symbol, list[Term]], sympy.Symbol | None]:
    """Compute device-index terms from stride_map and host index.

    Given a stride_map (mapping device dimensions to their strides) and a host
    index expression over loop variables, decompose the index into Terms,
    grouped by loop variable (traversal dimension) instead of device dimension.

    Returns one list[Term] per traversal (loop) variable in var_ranges iteration
    order, each containing all Terms this variable contributes to (one per
    device dimension it touches), answering "what does loop variable v
    contribute to the device address" directly.

    Args:
        var_ranges: dict of loop variable -> range, in traversal order
                   (e.g. {p0: 2, p1: 4096})
        device_size: device dimensions' extents (e.g. [2, 64])
        stride_map: stride per device dimension (e.g. [64, 1])
        index: host index expression over vars (e.g. p0*64 + p1)

    Returns:
        (grouped, stick_var): grouped is dict[var -> list[Term]], keyed by
        traversal variable in var_ranges iteration order, each var mapping to
        an ordered list of Terms (one per device dim this var touches, in device
        dim order 0..n-1). stick_var is the loop variable owning the last
        device dimension's term, or None if that dimension carries no variable.

    Raises:
        DeviceIndexError: if index structure violates device-index invariants.
    """
    assert all(isinstance(s, int) for s in device_size), (
        f"compute_device_index_terms requires concrete device_size, got {device_size}"
    )
    assert all(isinstance(s, int) for s in stride_map), (
        f"compute_device_index_terms requires concrete stride_map, got {stride_map}"
    )

    n = len(device_size)
    assert len(stride_map) == n, (
        f"stride_map length {len(stride_map)} != device_size length {n}"
    )

    # Compute terms for each device dimension (like compute_coordinates does).
    device_terms: list[Term | None] = [None] * n

    # Build next_stride[i] = smallest stride > stride[i]
    next_stride = [math.inf] * n
    for i in range(n):
        for j in range(n):
            if stride_map[j] > stride_map[i] and stride_map[j] < next_stride[i]:
                next_stride[i] = stride_map[j]

    # Process each variable in the index.
    for var in index.free_symbols:
        # Extract the coefficient and step for this variable.
        step_expr = index.xreplace({var: 1}).xreplace(
            {v: 0 for v in index.free_symbols if v != var}
        )
        step_expr_int = (
            int(step_expr) if isinstance(step_expr, (int, sympy.Integer)) else step_expr
        )

        # For each device dimension, check if this var's stride range includes it.
        for dim in range(n):
            if device_size[dim] == 1:
                continue  # Skip size-1 dims
            st = stride_map[dim]
            # var contributes to this dim if st <= step < next_stride[dim].
            if st <= step_expr_int < next_stride[dim]:
                # Compute den/mod for this term.
                if next_stride[dim] < math.inf and step_expr_int >= next_stride[dim]:
                    # Overflow: var wraps across multiple dims
                    den = next_stride[dim] // st
                    mod = next_stride[dim]
                else:
                    # No overflow: simple term
                    den = step_expr_int // st
                    mod = step_expr_int

                # Create Term: num=1, var, den, mod, offset=0
                device_terms[dim] = Term(
                    num=sympy.S.One,
                    den=sympy.Integer(den),
                    var=var,
                    mod=sympy.Integer(mod),
                    dim_size=sympy.Integer(device_size[dim]),
                    offset=sympy.S.Zero,
                )

    # Group terms by variable, preserving var_ranges iteration order.
    # Initialize dict with all vars from var_ranges in iteration order.
    grouped: dict[sympy.Symbol, list[Term]] = {var: [] for var in var_ranges}
    stick_var = None
    for dim in range(n):
        term = device_terms[dim]
        if term is not None:
            grouped[term.var].append(term)
            if dim == n - 1:  # Last device dimension is the stick
                stick_var = term.var

    # Filter out empty lists to keep only vars that actually contributed.
    return (
        {var: terms for var, terms in grouped.items() if terms},
        stick_var,
    )


def validate_device_index_terms(
    grouped: dict[sympy.Symbol, list[Term]], stick_var: sympy.Symbol | None
) -> None:
    """Validate device-index term structure invariants.

    Checks:
    - No traversal variable has repeated (den, mod) pairs (no repeated residues)
    - Non-stick residue terms' den evenly divides the stick size

    Args:
        grouped: dict[var -> list[Term]] from compute_device_index_terms
        stick_var: the variable owning the stick dimension, or None

    Raises:
        DeviceIndexError: if any invariant is violated.
    """
    # Extract stick size from the stick term (last term of stick_var's list).
    stick_size = None
    if stick_var is not None and stick_var in grouped:
        terms = grouped[stick_var]
        if terms:
            stick_size = int(terms[-1].dim_size)

    # Check each variable for repeated residues and residue den divisibility.
    for var, terms in grouped.items():
        is_stick_var = var is stick_var
        seen_keys: set[tuple] = set()

        for idx, term in enumerate(terms):
            is_stick_term = is_stick_var and idx == len(terms) - 1
            den_int = (
                int(term.den) if isinstance(term.den, (int, sympy.Integer)) else None
            )
            mod_int = (
                int(term.mod) if isinstance(term.mod, (int, sympy.Integer)) else None
            )
            if den_int is None or mod_int is None:
                raise DeviceIndexError(
                    message="term den/mod must be concrete integers",
                    var=var,
                    details=f"den={term.den}, mod={term.mod}",
                )
            key = (den_int, mod_int)

            # Check for repeated (den, mod) pair.
            if key in seen_keys:
                raise DeviceIndexError(
                    message="traversal variable has repeated residue",
                    var=var,
                    details=f"repeated (den, mod)={key}",
                )
            seen_keys.add(key)

            # Check residue den divides stick size (for non-stick terms).
            if not is_stick_term and stick_size is not None:
                den = den_int
                if stick_size % den != 0:
                    raise DeviceIndexError(
                        message="residue term den is not a multiple of stick size",
                        var=var,
                        details=f"den={den}, stick_size={stick_size}",
                    )
