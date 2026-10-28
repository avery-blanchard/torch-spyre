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

from collections.abc import Sequence

import sympy

from .views import Term, compute_coordinates, normalize_coordinates


class DeviceIndexError(Exception):
    """Raised when device-index term structure violates an invariant."""

    def __init__(
        self,
        message: str,
        var: sympy.Symbol | None = None,
        details: str | None = None,
    ) -> None:
        self.message = message
        self.var = var
        self.details = details
        super().__init__(str(self))

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

    # Delegate coordinate extraction to compute_coordinates, which correctly
    # handles FloorDiv/Mod terms, overflow, and all index structures that arise
    # in practice.  normalize_coordinates then converts the raw sympy coordinate
    # expressions into structured Term objects (num, den, var, mod, dim_size).
    _synthetic_idx = [0]

    def _synthetic_var() -> sympy.Symbol:
        v = sympy.Symbol(f"_si{_synthetic_idx[0]}")
        _synthetic_idx[0] += 1
        var_ranges[v] = 1
        return v

    try:
        coords = compute_coordinates(device_size, stride_map, var_ranges, index)
    except Exception as e:
        raise DeviceIndexError(
            message="compute_coordinates failed",
            details=str(e),
        ) from e

    try:
        terms = normalize_coordinates(var_ranges, device_size, coords, _synthetic_var)
    except Exception as e:
        raise DeviceIndexError(
            message="normalize_coordinates failed",
            details=str(e),
        ) from e

    # Group terms by their loop variable (traversal dimension), preserving
    # var_ranges iteration order.  Terms with var=None are constant/synthetic
    # and do not contribute to the grouping.
    grouped: dict[sympy.Symbol, list[Term]] = {var: [] for var in var_ranges}
    for term in terms:
        if term.var is not None and term.var in grouped:
            grouped[term.var].append(term)

    # The stick variable is the var owning the last term (stick dimension).
    stick_var = terms[-1].var if terms and terms[-1].var is not None else None

    return (
        {var: term_list for var, term_list in grouped.items() if term_list},
        stick_var,
    )


def validate_device_index_terms(
    grouped: dict[sympy.Symbol, list[Term]],
    stick_var: sympy.Symbol | None,
    num_elements_per_stick: int,
) -> None:
    """Validate device-index term structure invariants.

    Checks (per the device tensor spec):
    - No traversal variable has repeated (den, mod) pairs (no repeated residues)
    - Nums in the residue (non-stick terms) must all be multiples of
      num_elements_per_stick

    Args:
        grouped: dict[var -> list[Term]] from compute_device_index_terms
        stick_var: the variable owning the stick dimension, or None
        num_elements_per_stick: element count per stick for the tensor's dtype

    Raises:
        DeviceIndexError: if any invariant is violated.
    """
    for var, terms in grouped.items():
        is_stick_var = var is stick_var
        seen_keys: set[tuple] = set()

        for idx, term in enumerate(terms):
            is_stick_term = is_stick_var and idx == len(terms) - 1
            num_int = (
                int(term.num) if isinstance(term.num, (int, sympy.Integer)) else None
            )
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

            # Spec: nums in the residue must be multiples of num_elements_per_stick.
            # num=1 means the term steps by one full row (always a whole number of
            # sticks by construction); only check terms with num > 1, which encode
            # a sub-row stride that must still land on a stick boundary.
            if not is_stick_term and num_int is not None and num_int > 1:
                if num_int % num_elements_per_stick != 0:
                    raise DeviceIndexError(
                        message="residue term num is not a multiple of num_elements_per_stick",
                        var=var,
                        details=f"num={num_int}, num_elements_per_stick={num_elements_per_stick}",
                    )
