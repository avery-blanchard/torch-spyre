# Multi-Core Work Division for Indirect Access

## Summary

Indirect access reads or writes rows of a table at positions chosen at runtime
by an index tensor:

* **gather** (`out = x[i]`) reads rows from a value table `x`;
* **scatter** (`out[i] = src`) writes rows into a destination table `out`.

This page describes running both across multiple Spyre cores **correctly** under
work division.

The single load-bearing rule, in both directions, is: **the work-division
planner must never split along one of the shared indirect table's data
dimensions.** That table — the value table for a gather, the destination table
for a scatter — is shared at a single base across cores: every core must be able
to address any row. Slicing one of its data dimensions (e.g. the hidden dim of
an embedding, the head dim of a KV cache) per core silently returns wrong
results — every core addresses column 0 of the shared table.

The fix makes such ops correct. It does **not** parallelise the data dimensions
themselves; that is future work (see
[Limitations](#limitations-and-future-work)).

Scatter carries **one extra condition** gather does not: because the
data-dependent index sits on the *output*, parallelising the index-entry
dimension is only safe when no two cores target the same destination row. We
enable it for **overwrite** scatters and leave **accumulating** scatters serial.

## Background: how indirect access is compiled

### Gather — indirect on a load

A gather is lowered to a **Pointwise `identity` op** with three tensor
arguments:

| Arg | Role | Device coordinates |
|---|---|---|
| index | the positions to read, tagged `KERNEL_IDX` | regular, statically known |
| value | the table being read from (shared) | carries an `IndirectAccess(name)` node at the gathered dimension |
| output | the gathered result | regular, statically known |

The indirection lives in a **load**: `inner_fn` reads the value table at a
runtime-computed index.

### Scatter — indirect on a store

A scatter (`out[i] = src`) is lowered to a **`Scatter` IR node** whose
`inner_fn` reads the source **directly**; the data-dependent destination lives
in `Scatter.output_indexer`:

```python
def inner_fn(index):
    i0, i1, i2 = index
    tmp0 = ops.load(arg2_1, i2 + 512*i1 + 32768*i0)   # reads src directly
    return tmp0
# the indirection is in Scatter.output_indexer, keyed on the `indices` closure
```

Its three tensor arguments mirror the gather, with the roles of value and output
swapped:

| Arg | Role | Device coordinates |
|---|---|---|
| index | the positions to write, tagged `KERNEL_IDX` | regular, statically known |
| src (input) | the values being written | regular — per-core base advances with the split |
| output | the destination table (shared) | carries an `IndirectAccess(name)` node at the scattered dimension |

### The shared model

In both cases the `IndirectAccess(name)` node (see `op_spec.IndirectAccess`)
marks the shared table's **indirect axis** — the dimension whose row is selected
at runtime by the index. That axis has **no iteration-space symbol**; its address
is resolved on the device via `SEGMENT_OFFSETS`, with `maxDimSizes == 1` for that
dimension.

Work division splits the op's iteration space across cores. For a gather over
`x : [M, K, N]` with `i : [Q]`, the iteration space is `{d0 = Q, d1 = K,
d2 = N}` and the output is `[Q, K, N]`. The value's gather axis `M` is **not** an
iteration variable — the iteration dims are the index-entry dim (`Q`) and the
value's data dims (`K`, `N`). A scatter over `out : [M, K, N]` with `i : [Q]` has
the same iteration space `{d0 = Q, d1 = K, d2 = N}`; the destination's scatter
axis `M` is the indirect axis.

### The gather already parallelises along the index — by construction

`get_mem_deps_from_rw`
([pass_utils.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/pass_utils.py))
filters indirect reads out of work division's inputs, so the planner only ever
sees the **index read** and the **output write** — both regular tensors. It
naturally distributes cores over their dimensions, which include the index-entry
dim. So index-driven parallelism works for a gather without special handling.

For a scatter : its index-entry dim is **absent** from the
destination's direct coordinates (the row is data-dependent), so it needs an
explicit nudge to become a split axis — see
[fix #2](#2-parallelise-over-the-index-entry-dim) below.

## The defect

The planner ranks dimensions by size and splits the largest. When a shared
table's data dim (`K` or `N`) is the largest — the common case for **wide rows**
(embedding `hidden`, expert weight matrices, attention `head_dim`) — it splits
that dimension. That is wrong, because the table is **shared**: every core must
address any row, so it cannot be sliced per core.

Concretely, under a `K`-split the regular tensor advances its per-core base
across `K` (core `c` uses columns `[2c, 2c+2)`), but the shared table's per-core
base does **not** advance. So for a gather every core reads value columns
`[0, 2)`, and for a scatter every core writes destination columns `[0, 2)`:

```text
gather:   out[:, 2c : 2c+2, :] = value[:, 0 : 2, :]    for every core c
scatter:  out[idx[...], 0 : 2, :] = src[..., 2c : 2c+2, :]    for every core c
```

Core 0 is correct by coincidence; the others use core-0's columns.

## The scatter-only correctness condition: index uniqueness

Splitting the index-entry dim `Q` hands each core a disjoint set of source rows /
index entries. For a gather the dimension being split is the *output*, so the
slices are trivially disjoint. For a scatter the destination rows are `idx[j]` —
*data-dependent*. Disjoint `Q`-ranges map to disjoint destination rows **if
`idx` is injective**. If two entries on different cores collide
(`idx[j1] == idx[j2]`), they race. Gather has no analogue — reads never conflict.

We therefore enable the index-entry split for **overwrite** scatters
(`Scatter.scatter_mode is None`, e.g. `index_put`): PyTorch already leaves
duplicate-index overwrites non-deterministic, so a per-core split stays within
that contract. **Accumulating** scatters (`atomic_add`, e.g. `scatter_add_`)
would need atomic writes to remain correct under duplicate indices, so they are
left for a single core.

## The proposed fix

Five changes, in order of importance.

### 1. Forbid splitting the shared table's data dims — the correctness fix

Both directions are unified behind `collect_shared_indirect_tds`
([work_division.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/work_division.py)),
which returns every shared indirect tensor of an op with `IndirectAccess`-aware
coordinates:

* gather — each indirect value *read*, via `collect_indirect_value_tds`;
* scatter — the indirect destination *write*, built by `_build_output_td`
  applying `indirect_store_subs_from_op` so the row axis becomes an
  `IndirectAccess` and only its data dims remain as coordinate symbols.

`shared_indirect_data_syms` then takes the non-`IndirectAccess` coordinate
symbols of those tensors — the same extraction for both directions. The split
passes consult it through `indirect_forbidden_split_syms`, which combines it with
the partial-last-stick rule ([fix #5](#5-stick-align-the-index-entry-split-partial-last-stick));
`_default_split` removes the resulting symbols from the output and reduction
priority lists, so the core budget is distributed only over the index-entry
dims — **divide by the index, not the table.**

When the index dim cannot absorb all the cores, the op falls back to fewer cores
rather than splitting a data dim. **Correct-but-serial is the intended
trade-off; silent corruption is not.**

### 2. Parallelise over the index-entry dim

A gather's index-entry dim is already an output coordinate, so it is a split axis
by default. A scatter's index-entry dim is **absent** from the destination's
direct coordinates (the row is data-dependent), so `prioritize_dimensions`
classifies it as a *reduction* dim, which a non-reduction op never splits.
`indirect_store_entry_syms` returns those entry dims (the iteration symbols left
after removing the data dims), and `_default_split`'s `force_output_syms`
promotes them to output priority so the distributor splits them.

The split round-trips through the existing coefficient encoding without new
machinery: a scatter's entry dim has coefficient 0 in the (indirect) write index,
so `splits_by_index_coeff` encodes it via the first **non-indirect** read — the
direct `src` load selected by `_first_non_indirect_read_index` — and
`apply_splits_from_index_coeff` decodes it the same way at codegen.

This is gated on the [uniqueness condition](#the-scatter-only-correctness-condition-index-uniqueness):
`indirect_store_entry_syms` returns dims only for overwrite scatters.

### 3. Shared-table span guard (defensive)

The shared table's coordinates carry `IndirectAccess` (via
`collect_indirect_value_tds` for a gather, `_build_output_td` for a scatter), so
it is visible to the per-core span check with `get_per_core_span` treating an
`IndirectAccess` coordinate as contributing its full device extent (any core may
touch any row) and never splitting it. A gather's value table is pulled in as an
extra TensorDep (it is not in `args`); a scatter's destination is the output
TensorDep, already covered. `warn_if_per_core_overflow` then logs a
critical message if the table would exceed the documented 256 MB per-core span.

This is conservative: a 512 MB table read correctly in testing, so the limit's
applicability to indirect tables is not firmly established. The guard surfaces a
potential hardware-limit violation as a compile-time warning rather than failing.

### 4. Deterministic split round-trip

The split plan is encoded with the coefficients of the read/write index
expressions. An indirect read carries data-dependent symbols whose coefficients
are not a stable identity key, so the encode side (`apply_splits`) and both
decode sites (`work_distribution_pass`, `create_op_spec`) prefer the first
**non-indirect** read as the reduction-split reference index, via
`_first_non_indirect_read_index`. This same reference also carries a scatter's
entry-dim split (fix #2).

### 5. Stick-align the index-entry split (partial last stick)

Enabling the index-entry split (fixes #1–2) exposes a second hazard when the
entry count is **not a whole multiple of the index stick** (32 int32 entries per
128-byte stick). Work division splits the entry dim in whole sticks, so an
*even* per-core slice of a partial last stick — e.g. `Q = 40` = one full stick +
8 — hands the second core a slice that **straddles the index stick boundary**.
The backend cannot step a sticked dimension across a stick boundary *within* a core,
so the entries past the boundary are read from the wrong device addresses (the
result miscompiles on exactly the rows in the partial second stick).

The fix pads the **gather output**'s entry-dim `device_size` up to the next stick
multiple (`enforce_indirect_access_layout._pad_output_for_stick_aligned_split`,
e.g. `40 → 64`), leaving the logical size unchanged. The physical allocation
grows, so the per-core base becomes stick-aligned — matching the index tensor,
whose device layout is already stick-padded — and the D2H copy extracts the
logical rows from the larger allocation (its `physical_exceeds_logical` path). To
keep the per-core base stride consistent, `superdsc` grows the SDSC iteration to
the padded size **before** `_create_sdsc_tensors` computes strides; otherwise the
base lands element-aligned (mid-stick) and the split still miscompiles.

`indirect_forbidden_split_syms`
([work_division.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/work_division.py))
enforces the invariant, **forbid unless provably padded**:

* it **forbids** a partial-last-stick entry dim by default — reading the index's
  *unpadded* logical count (`d.ranges`), which the output padding never changes;
* it **lifts** that forbiddance only when the output was grown to a whole stick,
  detected via `indirect_entry_output_dim(op).out_extent % eps == 0`.

So a padded gather splits stick-aligned, while anything unpadded falls back to a
single core rather than miscompiling. A **scatter cannot be padded**: its
destination is written in place (a mutation layout the pass skips) and its entry
row is data-dependent (an `IndirectAccess` coord), so `indirect_entry_output_dim`
returns `None`, the forbiddance is never lifted, and a partial-stick scatter
stays on a single core — correct-but-serial, never miscompiled.

Stick-aligned counts (a multiple of 32) are unaffected: no padding, the
forbiddance never fires, and they split exactly as before.

## Detection: load side vs. store side

Both directions discover their indirect symbols before scheduling via
`indirect_access_subs_from_op`
([pass_utils.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/pass_utils.py)),
which merges:

* `_build_indirect_load_subs` — re-executes `inner_fn` with `_IndirectIndexFinder`
  to learn which buffer's **load** produced each indirect index (gather);
* `_build_indirect_store_subs` — recovers the scatter index buffer from the
  `Scatter.output_indexer` closure (via `_find_scatter_index_buf_names`) and maps
  each indirect symbol in the **write** dep to it (scatter).

Both map an indirect symbol to `IndirectAccess(name)`, which `device_coordinates`
substitutes into the shared table's coordinates.

## Worked examples

### Gather

`out = x[i]` with `x : [128, 64, 512]`, `i : [256]`, `SENCORES=32`. Iteration
space `{d0 = Q = 256, d1 = K = 64, d2 = N = 512}`; the value's gather axis
`M = 128` is addressed by `IndirectAccess`. The index stickifies to `256 / 32 =
8` sticks on `d0`.

| | Without the fix | With the fix |
|---|---|---|
| Largest dim | `K = 64` | (forbidden) |
| Split | `K`, 32 cores | `d0` (index), 8 cores |
| Value tensor | per-core column base diverges | shared at base 0 |
| Result | wrong (every core reads column 0) | correct |

### Scatter

`out[i] = src` with `out, src : [5, 64, 512]`, `i : [5]` (a permutation),
`SENCORES=32`. Iteration space `{d0 = Q = 5, d1 = K = 64, d2 = N = 512}`; the
destination's scatter axis `M = 5` is addressed by `IndirectAccess`. The index
stickifies `d0` to `ceil(5 / 32) = 1` stick.

| | Without the fix | With the fix |
|---|---|---|
| Largest dim | `K = 64` | (forbidden) |
| Split | `K`, 32 cores | `d0` (index), 1 core (Q stickifies to 1) |
| Destination | per-core column base pinned, all cores write columns `[0,2)` | shared at base 0, row from `IndirectAccess` |
| Result | wrong / backend abort | correct (serial — index too small to split) |

In both cases parallelism is set by the index size in sticks:
`cores = core_split(ceil(Q / 32), SENCORES)` for a 1-D index — the largest
divisor of the index's stick count that fits the core budget (`Q = 256 → 8
sticks → 8`, `Q = 1024 → 32`; a partial count like `Q = 40` pads to `2` sticks →
`2` cores ([fix #5](#5-stick-align-the-index-entry-split-partial-last-stick)),
and a non-power-of-two `SENCORES = 6` rounds `8` sticks down to `4`). A spatial
(non-stick) index dimension splits directly. A small index (the scatter `Q = 5`
here) runs correct-but-serial.

## Validation

### Gather / scatter multicore scenarios

`examples/gather_multicore_exp.py` (gather) and `examples/scatter_multicore_exp.py`
(scatter) each run three scenarios at the shapes above, isolating one behavior by
changing only the table contents:

| Scenario | Table contents | Isolates | With fix | Without fix |
|---|---|---|---|---|
| `realistic_random` | `rand()` | real-world indirect access | pass | fail (~97% wrong) / abort |
| `column_addressing` | `[r,k,n] = k` | K-column integrity | pass  | fail (all use col 0) |
| `row_addressing` | `[r,k,n] = r` | indirect-axis selection | pass | pass (axis never split) |

`row_addressing` passes either way and is included to show the indirect axis is
always correct and hides the data-dim defect; `column_addressing` and
`realistic_random` are the correctness signal.

```bash
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 SENCORES=32 python examples/gather_multicore_exp.py
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 SENCORES=32 python examples/scatter_multicore_exp.py
```

**Note:** `examples/scatter_multicore_exp.py` is a **work in progress** — the
scatter backend path is not yet complete, so this example is illustrative and may
not pass end-to-end. `examples/gather_multicore_exp.py` is the validated one.

### Paged-attention examples

Two additional examples demonstrate indirect access in the paged-attention pattern
that motivated this feature:

* **`examples/paged_attention_compute.py`** — 1-D indirect access: a single
  gather reads KV-cache rows selected by a 1-D slot-index tensor. Runs the CPU
  reference and the Spyre compiled path side-by-side and validates the outputs.

* **`examples/paged_attention_kernel.py`** (**work in progress**) — full paged
  attention with combined gather (KV read) and scatter (KV write): populates a
  paged KV cache with scatter (`cache[slot_ids] = new_kv`), reads it back with
  gather (`kv = cache[slot_ids]`), and runs the attention computation. Because it
  depends on the scatter backend path (not yet complete), this example is
  illustrative and may not pass end-to-end yet; `paged_attention_compute.py`
  (gather only) is the validated one.

Note: these workloads have a small block-table index and wide KV rows, so they
exercise the correctness fix but typically run on 1–2 cores (see
[Limitations](#limitations-and-future-work)).

```bash
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 SENCORES=8 python examples/paged_attention_compute.py
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 SENCORES=8 python examples/paged_attention_kernel.py
```

## Limitations and future work

* **Parallelism is capped by the index size.** The current implementation
  parallelises only over the index dimension. Workloads with a small, fixed index
  but wide rows — MoE expert gathers (top-k routing), paged-attention KV reads
  (short block tables), wide embedding lookups at small batch — get few cores
  (often 1–2). They are **correct**, but not maximally parallel.

* **Accumulating scatters run serially.** `scatter_add_` / `index_put(...,
  accumulate=True)` are left on a single core because a multi-core split would
  need atomic accumulate to stay correct under duplicate indices. Enabling them
  requires backend atomic-add support.

* **Multi-index scatter is not divided.** When the destination index is built
  from more than one index tensor, `_build_indirect_store_subs` cannot
  unambiguously map each indirect symbol to its source buffer, so it returns no
  subs and the op stays unsplit (correct, serial).

* **Parallelising the data dims is deferred.** A prototype let the planner split
  a data dim and advanced the table's per-core base along it. On hardware the
  per-core base of a `value_tensor` allocation is interpreted as an absolute
  offset into the table *where the indirect access happens*, so advancing it
  shifts the addressed **row**, not the **column** — there is no per-core base
  that expresses "different columns per core." Making the data-dim split correct
  requires carrying the per-core column offset in the coordinate folds and/or a
  backend change to how `value_tensor` bases are interpreted. This needs the
  DeepTools/backend team and is out of scope here.

## Implementation

| File | Change |
|---|---|
| `_inductor/pass_utils.py` | `_build_indirect_load_subs`, `_build_indirect_store_subs`, `_wrap_indirect_subs`, `indirect_access_subs_from_op` (merges both), `indirect_store_subs_from_op`, `_first_non_indirect_read_index`, `indirect_entry_output_dim` + `IndirectEntryDim` (partial-stick, fix #5) |
| `_inductor/work_division.py` | `collect_shared_indirect_tds` (gather reads + scatter destination), `shared_indirect_data_syms`, `indirect_forbidden_split_syms` (shared-table dims + partial-stick rule, fix #5), `_non_indirect_coord_syms`, `_build_output_td`, `collect_indirect_value_tds`, `indirect_store_entry_syms`, `forbidden_split_syms` + `force_output_syms` in `_default_split`, `IndirectAccess` span guard |
| `_inductor/enforce_indirect_access_layout.py` | `_pad_output_for_stick_aligned_split` — grows a partial-stick gather output's entry-dim `device_size` to a whole stick (fix #5) |
| `_inductor/codegen/superdsc.py` | grow the SDSC index-entry iteration to the padded output size before `_create_sdsc_tensors` so the per-core base is stick-aligned (fix #5) |
| `_inductor/spyre_kernel.py` | non-indirect read index in `create_op_spec` |
| `examples/gather_multicore_exp.py` | three-scenario multicore validation example (gather) |
| `examples/scatter_multicore_exp.py` | three-scenario multicore example (scatter) — **WIP** |
| `examples/paged_attention_compute.py` | paged-attention 1-D gather example (CPU + Spyre, validates outputs) |
| `examples/paged_attention_kernel.py` | full paged-attention gather + scatter example (KV write, read, attention) — **WIP** |

## See also

* [Work Division Planning](work_division_planning.md) — the three-pass planner
  this builds on.
