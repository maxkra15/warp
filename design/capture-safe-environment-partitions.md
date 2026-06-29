# Capture-Safe Environment-First Space Partitions

**Status**: Implemented

**Issue**: [GH-1407](https://github.com/NVIDIA/warp/issues/1407)

## Motivation

Environment-first FEM space partitions order nodes by environment and expose
per-environment offsets for independent numerical solves. This is the basis for
colocated multi-environment simulations such as Newton's implicit MPM solver,
whose multi-world requirement is tracked in
[newton-physics/newton#1074](https://github.com/newton-physics/newton/issues/1074).

The current `EnvironmentSpacePartition.rebuild()` implementation copies the
compressed environment offsets to NumPy so Python can determine the exact live
node count and allocate exact-size arrays. A device-to-host read synchronizes the
stream and is forbidden during CUDA graph capture. Consequently, rebuilding an
environment-first partition inside an outer CUDA graph fails even when the
caller supplies `max_node_count`, whose documented purpose is to avoid
device/host synchronization.

Other FEM partitions already use a fixed-capacity contract when a maximum count
is supplied. Extending that contract to environment-first partitions removes the
host dependency while retaining dynamic active-node reconstruction on every graph
replay. This enables fixed-grid, isolated multi-world implicit MPM steps to use
the same outer CUDA graph workflow as their single-world counterparts.

## Requirements

| ID | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1 | Rebuild an environment-first partition without device-to-host synchronization when `max_node_count >= 0` | Must | Includes construction during capture and rebuilding an existing partition |
| R2 | Keep allocation sizes, launch dimensions, and array pointers stable across graph replays | Must | Required by captured downstream commands |
| R3 | Recompute active node ordering, inverse mappings, and environment offsets on every replay | Must | Particles and active cells may move |
| R4 | Preserve environment-major ordering and independent environment batches | Must | No cross-environment coupling |
| R5 | Make `env_offsets` cover every fixed-capacity degree of freedom | Must | Required by `LinearOperator` batch semantics |
| R6 | Preserve the existing exact-size behavior when `max_node_count < 0` | Must | Includes host synchronization and environmentless-node handling |
| R7 | Preserve deterministic prefix truncation when capacity is below the live node count | Must | Existing `max_node_count` behavior |
| R8 | Support empty environments, zero capacity, underfilled capacity, exact fit, and overflow | Must | Offsets remain monotonic in every case |
| R9 | Demonstrate end-to-end outer CUDA graph replay in Newton's isolated fixed-grid implicit MPM solver | Must | Eager and captured trajectories must agree |
| R10 | Avoid new required dependencies and native-code changes | Should | The fix belongs in Warp FEM's Python/device-kernel layer |

**Non-goals**:

- Making uncapped, exact-size partition construction capture-safe. Exact dynamic
  allocation requires a host-visible count or a different allocation model.
- Capturing Newton's dense-grid bounds construction.
- Capturing NanoVDB topology construction for Newton's sparse MPM grid.
- Making Newton's CG, CR, or GMRES rheology paths outer-capturable. Those paths
  contain independent host reads for tolerance and result reporting.
- Introducing per-environment capacity parameters or changing the public
  `make_space_partition()` signature.
- Optimizing fixed-capacity matrix memory use; in-place sparse-matrix compression
  is an orthogonal improvement.

## Design

### Approach

When `max_node_count >= 0`, `EnvironmentSpacePartition` will use a fixed-capacity
representation analogous to `NodePartition`:

- The capacity is `min(max_node_count, space_topology.node_count())`.
- `node_count()`, `owned_node_count()`, and `interior_node_count()` return that
  capacity.
- `space_node_indices()` has exactly that capacity.
- Active nodes occupy an environment-major prefix. Remaining entries are inert
  capacity fillers drawn from the compressed environmentless suffix.
- For a non-whole geometry partition, only active prefix entries are installed in
  the global-to-partition mapping; filler entries remain `NULL_NODE_INDEX`.
- For a whole geometry partition, existing inclusion semantics are preserved:
  entries selected within the capacity remain mapped, including any unreferenced
  topology-node suffix.
- `env_offsets` always has `environment_count + 1` entries and its final entry is
  the partition capacity. If the live node count is below capacity, the unused
  suffix belongs to the final environment interval.

Assigning underfill to the final interval is a storage convention, not an active
node classification. The slots are not referenced by a non-whole partition's
cells and therefore contribute no assembled terms. Covering the suffix is
necessary because Warp's batched `LinearOperator` expects its offsets to
partition all scalar degrees of freedom; leaving the suffix uncovered would map
those entries to an invalid batch.

The uncapped path remains exact-size and unchanged. It may synchronize to read
the live count, and it continues returning `env_offsets is None` when a whole
partition contains environmentless nodes that are part of the exact partition.

### Device-Side Rebuild

The existing classification and compression stages remain responsible for
producing:

- `node_env`: one environment classification per topology node;
- `group_offsets`: device-resident prefix offsets for every environment plus the
  environmentless group; and
- `node_indices`: the environment-major permutation of topology-node indices.

`compress_node_indices()` remains generic and is not given partition-specific
capacity behavior. `EnvironmentSpacePartition.rebuild()` interprets its outputs
as follows in fixed-capacity mode:

1. Allocate or reuse `_node_indices` at the fixed capacity and copy the first
   `capacity` compressed indices into it.
2. Allocate or reuse `_space_to_partition` with one entry per topology node and
   fill it with `NULL_NODE_INDEX`.
3. Launch a device kernel that scatters the valid prefix into
   `_space_to_partition`. For non-whole partitions the valid prefix ends at
   `min(group_offsets[environment_count], capacity)`; for whole partitions it
   ends at `capacity`.
4. Allocate or reuse `_env_offsets` with `environment_count + 1` entries.
5. Launch a device kernel that writes
   `min(group_offsets[environment], capacity)` for intermediate offsets and
   writes `capacity` for the final offset.
6. Set the host-side `_node_count` to the already-known capacity and release
   temporary buffers without reading them.

All work after the host-known capacity calculation is a device operation. The
fixed-capacity branch performs no `.numpy()`, event synchronization, or
host-created offset upload.

### Single-Environment Fast Path

The current whole-geometry, single-environment fast path creates a fresh
two-entry offset array on each rebuild. It will instead allocate `_env_offsets`
once per compatible shape/device and update it in place to `[0, capacity]`.
This keeps the fast path pointer-stable and makes its rebuild safe to record and
replay.

### Object Lifetime and Cache Invalidation

For unchanged topology, device, environment count, and capacity, rebuild mutates
these arrays in place:

- `_node_indices`;
- `_space_to_partition`; and
- `_env_offsets`.

Pointer stability is a correctness property because captured kernels retain
array pointers while Python attribute assignment is not replayed. Structural
changes may replace arrays outside capture. When `_space_to_partition` is
replaced, the cached `partition_arg_value` must be invalidated so subsequent
launches cannot retain a stale mapping pointer.

FEM temporaries borrowed while either APIC or native CUDA graph capture is
active bypass `TemporaryStore`'s recycling pools. Capture-local allocations are
therefore owned by the captured graph or APIC recording rather than being made
available to an unrelated borrower while the graph still retains their
pointers. Persistent partition outputs remain owned by the partition object and
are reused in place.

The rebuild documentation will state that changing topology size, environment
count, device, or capacity is structural and is not supported inside an already
captured graph. Callers must reacquire arrays after such a change, matching the
existing API guidance.

### Capacity Underfill and Overflow

Underfill keeps the partition shape fixed. Active nodes remain at the front,
filler nodes remain unmapped for non-whole partitions, and the last environment
interval extends to capacity.

When the live node count exceeds capacity, the existing deterministic prefix is
retained. Environment offsets are clamped to capacity, nodes outside the prefix
remain unmapped, and the final offset equals capacity. This proposal does not add
a new exception or host-side overflow check because `max_node_count` currently
acts as a limit and existing callers and tests rely on truncation. Higher-level
applications should choose a conservative capacity; Newton derives its node
capacity from the maximum active-cell count and nodes per element.

### Newton Integration

The Newton change depends on the Warp change and does not introduce a
Newton-specific partition cache or whole-grid fallback. Its production solver
continues rebuilding active geometry and FEM spaces every step.

The supported outer-capture configuration is:

- isolated multi-world mode;
- a fixed MPM grid;
- a non-negative `max_active_cell_count`; and
- a graph-compatible nonlinear rheology solver (`auto`/Gauss-Seidel or Jacobi).

Capacity padding is safe for the nonlinear and collision paths:

- FEM restriction mappings leave inactive padding unreferenced.
- Collision rasterization initializes node positions to `OUTSIDE` and returns
  before environment lookup for inactive nodes.
- Padding contributes zero residual. Nonlinear convergence requires both the
  batch L2 condition and its maximum residual condition, so final-batch padding
  cannot cause premature convergence.

Linear Krylov configurations are not included in the support claim. They have
independent `.numpy()` calls and use offset-derived counts when choosing a shared
absolute tolerance. A future design may expose exact active environment counts
separately from capacity-covering batch offsets, but that metadata is unnecessary
for the nonlinear capture path.

The Newton dependency change is sequenced after the Warp PR. Local validation
uses the Warp worktree directly. Newton's lock file should be updated only when a
nightly or release containing the Warp fix is available; no temporary local-path
or feature-branch dependency will be committed.

### Alternatives Considered

#### Cache the Environment Partition in Newton

Rejected because the set of active grid nodes and the environment offsets can
change whenever particles cross cells. Replaying a cached mapping would silently
write into stale degrees of freedom.

#### Use the Whole Fixed Grid

Rejected because it replaces active-subset work with full-domain work. A small
two-world probe expanded velocity, strain, and collider node counts by roughly
three to four orders of magnitude. It also defeats the memory and compute
benefits of active partitions.

#### Add Per-Environment Capacities

Rejected for this change because it requires a new API and a segmented packing
algorithm. A single environment can overflow its reservation while total free
capacity remains available. A global capacity already matches the existing
partition contract and Newton's active-cell limit.

#### Add a Synthetic Padding Batch

Rejected because it changes the offset shape from `environment_count + 1` to
`environment_count + 2`, causing numerical solvers to observe an extra
environment. Assigning inert tail capacity to the final interval preserves the
public batch count.

#### Leave Tail Degrees of Freedom Outside `env_offsets`

Rejected because batched linear-solver kernels assume the last offset covers the
operator shape. An uncovered suffix can resolve to batch `-1` and produce invalid
array access.

## Testing Strategy

### Warp Tests

Extend the existing FEM multi-environment tests rather than adding a new test
module:

1. **Underfilled explicit partition**: verify fixed node count and array lengths,
   environment-major active prefix, inactive inverse mappings, monotonic offsets,
   and `env_offsets[-1] == capacity`.
2. **Exact fit and overflow**: verify stable prefix selection and clipped offsets.
3. **Empty environments**: cover leading, middle, and trailing empty ranges.
4. **Whole topology with an unused node**: preserve uncapped exact behavior and
   verify the documented fixed-capacity suffix convention.
5. **Single environment and zero capacity**: verify fast-path reuse and boundary
   behavior.
6. **Captured rebuild and replay**: capture `ExplicitGeometryPartition.rebuild()`
   followed by `EnvironmentSpacePartition.rebuild()`, then replay with at least
   two different cell masks. Verify that mappings and offsets change while array
   objects and pointers remain stable.
7. **Captured FEM integration**: assemble through an environment-first partition
   inside capture and compare values and sparsity with eager execution.
8. **Batched numerical smoke test**: use underfilled offsets in a batched reduction
   or `LinearOperator` operation to prove every capacity slot belongs to a valid
   batch.

Correctness cases run on CPU and CUDA where supported. Capture/replay cases run on
devices selected by Warp's graph-capture test utilities. Tests assert results and
pointer stability, never timing ratios.

### Newton Tests

Add one focused CUDA-only end-to-end regression to
`newton/tests/test_implicit_mpm.py`:

1. Build two coincident worlds with identical small particle blocks and opposite
   x velocities.
2. Use fixed-grid PIC, isolated worlds, Jacobi, a conservative active-cell
   capacity, zero convergence tolerance, and a fixed iteration budget.
3. Construct independent eager and captured solver/state pairs from identical
   initial data.
4. Capture two substeps so the graph contains both directions of the existing
   double-buffered state swap.
5. Replay multiple cycles while running the same number of eager substeps.
6. Verify that at least one particle crosses a voxel boundary and that an
   environment offset changes, proving the graph rebuilds active topology.
7. Compare positions, velocities, velocity gradients, stress, and strain between
   eager and replayed states; require finite values and appropriate tolerances.
8. Verify opposite per-world displacement/velocity signs, demonstrating that
   colocated worlds remain isolated under replay.

The test skips devices without the CUDA memory-pool and conditional-graph
capabilities required by the existing implicit rheology graph path.

### Documentation and Performance Validation

- Update Warp's `max_node_count` documentation to describe fixed-capacity
  semantics and final-environment padding.
- Add a Warp changelog entry describing capture-safe capped environment-first
  rebuilds.
- Update Newton's world documentation to state that isolated fixed-grid
  nonlinear MPM supports outer CUDA graph capture, while dense, sparse, and
  linear Krylov limitations remain.
- Add a Newton changelog entry for the restored capture capability.
- Retain the existing Newton multi-world ASV benchmark. Use a local eager versus
  graph-replay timing probe to quantify launch-overhead improvement, but do not
  add timing assertions or a second benchmark surface solely for this fix.

## Delivery and Merge Order

1. Land the Warp implementation, tests, documentation, and changelog.
2. Publish a Warp nightly or release containing the change.
3. Update Newton's lock/dependency to that artifact.
4. Land the Newton regression and documentation update.

The two changes remain separate and reviewable. The Newton PR may be developed
and validated against the local Warp worktree before the artifact exists, but it
must not commit a transient path dependency. Neither repository is pushed by the
implementation agent.
