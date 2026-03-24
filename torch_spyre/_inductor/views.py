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

# Helper methods to handle views

import math
import sympy
from typing import Optional, Sequence


def compute_relative_stride(
    rank: int, device_size: Sequence[sympy.Expr], dim_map: Sequence[int]
) -> list[sympy.Expr]:
    """
    Compute strides of device dimensions with respect to host dimensions
    """
    acc = [sympy.S.One] * rank
    rel_stride = [-1] * len(dim_map)
    for device_dim in range(len(dim_map) - 1, -1, -1):
        dim = dim_map[device_dim]
        if dim != -1:
            rel_stride[device_dim] = acc[dim]
            acc[dim] *= device_size[device_dim]
    return rel_stride


def compute_coordinates(
    size: Sequence[sympy.Expr],
    stride: Sequence[sympy.Expr],
    var_ranges: dict[sympy.Symbol, sympy.Expr],
    index: sympy.Expr,
) -> list[sympy.Expr]:
    """
    Compute an array of coordinate expressions from an index expression.

    Stride and index must be relative to the same storage (both host or device).
    Stride values<=0 are ignored.
    """
    # find stride immediately strictly larger that dim stride
    n = len(size)
    next_stride = [sympy.oo] * n
    for i in range(n):
        for j in range(n):
            # n^2 is ok since n is small
            if next_stride[i] > stride[j] and stride[j] > stride[i]:
                next_stride[i] = stride[j]
    # compute coordinate expressions
    coordinates = [sympy.S.Zero] * n
    vars = index.free_symbols
    for var in vars:
        if var_ranges[var] <= 1:
            continue  # ignore var with trivial range
        # isolate current var
        term = index.subs({v: 0 for v in vars - {var}})
        # compute index({var=1}) and index({var=var_ranges[var]})
        step = term.subs(var, 1)
        limit = term.subs(var, var_ranges[var])
        # find primary dim with largest stride less than or equal to step
        primary_stride = 0
        primary_dim = -1
        for dim in range(n):
            if size[dim] == 1:
                continue  # ignore dim with size 1
            st = stride[dim]
            if st <= step and st > primary_stride:
                # found candidate primary dim
                primary_stride = st
                primary_dim = dim
            elif st > step and st < limit:
                # var range intersects dim, add term
                if next_stride[dim] < limit:
                    # var range overflows dim
                    coordinates[dim] += var * step % next_stride[dim] // st
                else:
                    coordinates[dim] += var * step // st
        # add term for primary dim
        if next_stride[primary_dim] < limit:
            coordinates[primary_dim] += (
                # var range overflows primary dim
                var * step % next_stride[primary_dim] // primary_stride
            )
        else:
            coordinates[primary_dim] += var * step // primary_stride
    return coordinates


# deprecated: replace with compute_coordinates with stride_map
def compute_device_coordinates(
    size: Sequence[sympy.Expr],
    stride: Sequence[sympy.Expr],
    device_size: Sequence[sympy.Expr],
    dim_map: Sequence[int],
    var_ranges: dict[sympy.Symbol, sympy.Expr],
    index: sympy.Expr,
) -> list[sympy.Expr]:
    """
    Derive an array of coordinate expressions into a device tensor from an index
    """
    rel_stride = compute_relative_stride(len(size), device_size, dim_map)
    host_coordinates = compute_coordinates(size, stride, var_ranges, index)
    coordinates = [sympy.S.Zero] * len(device_size)
    for dim in range(len(device_size)):
        if dim_map[dim] == -1:
            continue
        expr = host_coordinates[dim_map[dim]]
        vars = expr.free_symbols
        for var in vars:
            term = expr.subs({v: 0 for v in vars - {var}})
            step = term.subs(var, 1)
            limit = term.subs(var, var_ranges[var])
            if limit > rel_stride[dim] and step < rel_stride[dim] * device_size[dim]:
                coordinates[dim] += term // rel_stride[dim]
    return coordinates


def matching_dim(coords: list[sympy.Expr], expr: sympy.Expr) -> Optional[int]:
    """
    Given a coordinate array and an expression, determine if there is a unique
    dimension in coords whose coordinate expression is exactly the one free variable
    in the expression.  Return None if expr does not have exactly one free variable
    or if there is not exactly one matching dimension in coords.
    """
    if len(expr.free_symbols) != 1:
        return None
    v = next(iter(expr.free_symbols))
    dims = [d for d, e in enumerate(coords) if e == v]
    if len(dims) != 1:
        return None
    else:
        return dims[0]


def normalize_coordinates(var_ranges, size, coordinates):
    results = []
    for coordinate, dim_size in zip(coordinates, size):
        expr = coordinate.replace(sympy.floor, lambda x: x)
        vars = expr.free_symbols
        if len(vars) == 0:
            results.append([1, None, None, None, dim_size])
        result = []
        for var in vars:
            term = expr.subs({v: 0 for v in vars - {var}})
            if term.is_symbol:
                result.append(
                    [sympy.S.One, sympy.S.One, var, var_ranges[var], dim_size]
                )
            elif term.func == sympy.Mod:
                result.append([sympy.S.One, sympy.S.One, var, term.args[1], dim_size])
            elif term.func == sympy.Mul and term.args[0].is_rational:
                expr0, expr1 = term.args
                mod = expr1.args[1] if expr1.func == sympy.Mod else var_ranges[var]
                result.append([expr0.numerator, expr0.denominator, var, mod, dim_size])
            else:
                raise IndexError
        result.sort()
        result.reverse()
        for r in result:
            r[-1] = dim_size // r[0]
            dim_size = r[0]
            results.append(r)

    new_results = []
    tmp = results[0]
    for result in results[1:-1]:
        if tmp[0] == 1 and tmp[1] == result[3] and tmp[2] == result[2]:
            tmp[0] = result[0]
            tmp[1] = result[1]
            tmp[4] *= result[4]
        else:
            if tmp[4] > 1:
                new_results.append(tmp)
            tmp = result
    if tmp[4] > 1:
        new_results.append(tmp)
    new_results.append(results[-1])
    return new_results


def align_tensors(var_ranges, tensors, op_it_space_splits={}):
    splits = {var: set() for var in var_ranges.keys()}
    breakdown = []
    stick_dim = []
    stick_size = []
    for t in tensors:
        intervals = normalize_coordinates(var_ranges, t["size"], t["coordinates"])
        stick_dim.append(intervals[-1][2])
        stick_size.append(intervals[-1][-1])
        breakdown.append(intervals)
        for num, den, var, mod, _ in intervals:
            if var is not None:
                if den != stick_size[-1] or var != stick_dim[-1]:
                    splits[var].add(den)
                if mod != stick_size[-1] or var != stick_dim[-1]:
                    splits[var].add(mod)
    splits = {var: sorted(val) for var, val in splits.items()}

    new_var_ranges = {}
    new_op_it_space_splits = {}
    n = 0
    remap = {}
    for var, split in splits.items():
        div = op_it_space_splits[var] if var in op_it_space_splits else 1
        if len(split) > 1:
            new_var_ranges[var] = split[1] // split[0]
            remap[var] = [var]
            for i in range(1, len(split) - 1):
                new_var = sympy.symbols(f"z{n}")
                n += 1
                new_var_ranges[new_var] = split[i + 1] // split[i]
                remap[var].append(new_var)
            for v in reversed(remap[var]):
                new_op_it_space_splits[v] = math.gcd(div, new_var_ranges[v])
                div //= new_op_it_space_splits[v]
        else:
            new_var_ranges[var] = var_ranges[var]
            new_op_it_space_splits[var] = (
                op_it_space_splits[var] if var in op_it_space_splits else 1
            )

    new_tensors = []
    for j, intervals in enumerate(breakdown):
        size = []
        coordinates = []
        for num, den, var, mod, dim_size in intervals:
            if var is None:
                size.append(dim_size)
                coordinates.append(sympy.S.Zero)
                continue
            if num * mod < dim_size:
                size.append(dim_size // num // mod)
                coordinates.append(sympy.S.Zero)
            if var == stick_dim[j] and den == stick_size[j]:
                for i in reversed(range(1, splits[var].index(mod))):
                    size.append(splits[var][i + 1] // splits[var][i])
                    coordinates.append(var)
                size.append(splits[var][1] // den)
                coordinates.append(var // den)
            elif var == stick_dim[j] and mod == stick_size[j]:
                size.append(mod)
                coordinates.append(var % mod)
            else:
                for i in reversed(
                    range(splits[var].index(den), splits[var].index(mod))
                ):
                    size.append(splits[var][i + 1] // splits[var][i])
                    coordinates.append(remap[var][i])
            if num > 1:
                size.append(num)
                coordinates.append(sympy.S.Zero)
        num, den, var, mod, dim_size = intervals[-1]
        new_tensors.append({"size": size, "coordinates": coordinates})

    rank = max([len(t["size"]) for t in new_tensors])
    for t in new_tensors:
        gap = rank - len(t["size"])
        t["size"] = [sympy.S.One] * gap + t["size"]
        t["coordinates"] = [sympy.S.Zero] * gap + t["coordinates"]

    for t in new_tensors:
        vars = t["coordinates"][-1].free_symbols
        if len(vars) == 1:
            stick_dim_var = next(iter(vars))
            found = False
            for i in range(len(t["coordinates"]) - 1):
                vars = t["coordinates"][i].free_symbols
                if stick_dim_var in vars:
                    found = True
                    continue
            if not found:
                for i in range(len(t["coordinates"]) - 1):
                    if t["size"][i] == 1:
                        t["coordinates"][i] = stick_dim_var // t["size"][-1]
                        t["coordinates"][-1] = stick_dim_var % t["size"][-1]
                        continue

    return new_var_ranges, new_tensors, new_op_it_space_splits


if __name__ == "__main__":
    from sympy import floor, Mod

    x0, x1, x2, x3, x4, x5, x6 = sympy.symbols("x0 x1 x2 x3 x4 x5 x6")

    print(
        align_tensors(
            {x0: 16384, x1: 256, x2: 30},
            [
                {
                    "size": [2, 128, 256, 64],
                    "coordinates": [
                        floor((Mod(x0, 128)) / 64),
                        floor(x0 / 128),
                        x1,
                        Mod(x0, 64),
                    ],
                },
                {
                    "size": [256, 1200, 64],
                    "coordinates": [floor(x0 / 64), floor(x2 * 2), Mod(x0, 64)],
                },
                {
                    "size": [256, 256, 64],
                    "coordinates": [floor(x0 / 64), x1, Mod(x0, 64)],
                },
            ],
        )
    )

    print(
        align_tensors(
            {x0: 128},
            [{"size": [2, 64], "coordinates": [x0 // 64, x0 % 64]}],
        )
    )

    print(
        align_tensors(
            {x0: 64},
            [{"size": [1, 64], "coordinates": [sympy.S.Zero, x0]}],
        )
    )

    print(
        align_tensors(
            {},
            [{"size": [1, 64], "coordinates": [sympy.S.Zero, sympy.S.Zero]}],
        )
    )

    print(
        align_tensors(
            {x0: 128, x2: 32, x1: 128},
            [
                {
                    "size": [128, 32, 2, 1, 1, 1, 64],
                    "coordinates": [
                        x1,
                        x2,
                        floor(x0 / 64),
                        sympy.S.Zero,
                        sympy.S.Zero,
                        sympy.S.Zero,
                        Mod(x0, 64),
                    ],
                },
                {
                    "size": [32, 128, 2, 1, 64],
                    "coordinates": [x2, x1, floor(x0 / 64), sympy.S.Zero, Mod(x0, 64)],
                },
            ],
        )
    )

    print(
        align_tensors(
            {x2: 2, x1: 256, x3: 4096, x0: 4096},
            [
                {
                    "size": [256, 32, 2, 2, 64],
                    "coordinates": [x1, x3 // 128, x3 % 128 // 64, x2, x3 % 64],
                },
                {
                    "size": [64, 4096, 64],
                    "coordinates": [x0 // 64, x3, x0 % 64],
                },
                {
                    "size": [256, 64, 2, 64],
                    "coordinates": [x1, x0 // 64, x2, x0 % 64],
                },
            ],
        )
    )

    print(
        align_tensors(
            {x0: 128, x1: 256},
            [
                {
                    "size": [16, 2, 16, 64],
                    "coordinates": [x1 // 16, x0 // 64, x1 % 16, x0 % 64],
                }
            ],
            {x0: 1, x1: 32},
        )
    )

    print(
        align_tensors(
            {x0: 128, x1: 256},
            [
                {
                    "size": [4, 2, 16, 4, 64],
                    "coordinates": [
                        x1 // 64,
                        x0 // 64,
                        x1 % 16,
                        x1 % 64 // 16,
                        x0 % 64,
                    ],
                }
            ],
            {x0: 1, x1: 32},
        )
    )
