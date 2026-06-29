# Rebuildable Nanogrid S2 Edge Topology Design

## Goal

Make degree-2 Serendipity (S2) spaces safe to reuse across rebuilds of a
rebuildable `warp.fem.Nanogrid`, including inside a CUDA graph. Preserve the
existing lazy behavior and cost for vertex-only spaces such as Q1.

## Problem

The NanoVDB rebuild merge request makes the cell and vertex index volumes
persistent. An S2 space also captures the Nanogrid edge index volume and its
capacity. Today that edge volume is a normal, active-sized `Volume`.
`Nanogrid._refresh_rebuildable_topology()` discards it after every cell rebuild,
leaving an existing `NanogridSpaceTopology` bound to a stale volume ID and stale
edge count.

The saved experimental patch demonstrates that rebuilding an edge volume in
place fixes the Newton MPM S2 case. It is not suitable for upstreaming as-is:
it reads obsolete private capacity fields and derives edge hierarchy capacity
with the vertex-specific factor eight. A source leaf can emit edges into as
many as twelve target leaf regions: four regions for each of three edge
orientations. The edge capacity bound therefore needs an edge-specific factor
of twelve.

## Selected Approach

Keep the change focused on edge-noded spaces. Add a small shared internal
helper for allocating rebuildable entity index volumes, while retaining
entity-specific candidate-generation kernels.

The helper will:

- Accept a fixed candidate buffer, its active mask, and an entity-specific
  hierarchy expansion factor.
- Read cell-grid capacity through the public `Volume.get_rebuild_info()` API.
- Reserve fixed active-voxel capacity equal to the candidate-buffer capacity.
- Bound leaf, lower, and upper node capacities by the candidate capacity and
  the corresponding cell-grid capacity multiplied by the expansion factor.

Vertex topology will use expansion factor eight; edge topology will use
expansion factor twelve. Sharing the allocator prevents their capacity policy
from drifting while keeping their coordinate-generation kernels explicit.

## Nanogrid Lifecycle

`Nanogrid` will continue to construct edge topology lazily:

1. A Q1 space never requests `edge_grid`, so it pays no edge memory or rebuild
   cost.
2. Creating an S2 space requests `edge_grid`. On a rebuildable Nanogrid this
   allocates a capacity-sized rebuildable edge volume and retains its candidate
   and mask buffers.
3. Every later `rebuild_topology_from_cells()` regenerates edge candidates from
   the capacity-sized cell buffer, masks its inactive tail, and rebuilds the
   same edge volume in place.
4. The edge `Volume` object and ID remain stable, so an existing S2 topology
   remains valid across CUDA graph replay.
5. `edge_count()` reports the fixed reserved edge-index capacity, keeping S2
   node-category offsets and field sizes stable.

Constructing new FEM spaces during graph capture remains unsupported, as it
already requires allocation. Required topology must be materialized before
capture.

Downstream consumers can feature-detect persistent edge support with
`getattr(fem.Nanogrid, "REBUILDABLE_EDGE_TOPOLOGY", False)` instead of
inferring it from `Volume` rebuildability or a Warp version.

## Scope Boundaries

This change intentionally does not make face topology rebuildable. S2 has only
vertex and edge nodes, so face topology is not required for the reported use
case. General Q2/Q3 or other face-noded basis support would also need persistent
face metadata, boundary flags, environment ownership, and capture-safe boundary
index compaction; that is a separate change.

The change also does not alter edge-key encoding, FEM partition semantics, or
Newton's warm-start policy. Those are independent concerns already present in
non-rebuildable Nanogrids.

## Failure Handling

Candidate masks exclude every capacity-tail cell at or beyond
`wp.volume_voxel_count(cell_grid)`. Capacity is derived conservatively from the
source grid's public rebuild metadata. The implementation will use the same
rebuild behavior as the MR's existing persistent vertex grid; no host
synchronization or allocation is introduced into the captured refresh path.

## Test Strategy

### Warp regression

Extend `test_nanogrid_rebuild_capture` to construct an S2 space before capture
and retain the original edge `Volume` object. The existing fixture rebuilds one
active cell into three active cells with one masked duplicate.

The test will assert:

- Before rebuild: cell, vertex, and edge active counts are `1`, `8`, and `12`.
- The S2 topology captures the original edge volume ID.
- After graph replay: active counts are `3`, `20`, and `32`.
- The Nanogrid still owns the same edge `Volume` object and ID.
- The pre-existing S2 topology remains bound to that ID.

This fails on the MR baseline because the held edge volume remains at twelve
active edges while the Nanogrid discards it.

Add a non-capture boundary case whose active cells cross a NanoVDB leaf boundary
to exercise the twelve-region edge hierarchy bound.

### Warp verification

Run the focused regression on every selected CUDA device with memory-pool
support, then the complete FEM geometry test module and pre-commit checks on the
changed files.

### Newton end-to-end verification

Run the Newton lava MPM example with the worktree Warp package, a rebuildable
sparse grid, S2 collider basis, and CUDA graph enabled. Require successful graph
capture, repeated replay without CUDA or capacity errors, and completion of the
configured steps. Run the same short scenario eagerly as a control.

## Alternatives Considered

### Edge and face topology in one change

This would make a stronger general high-order claim, but face topology owns
additional dynamic metadata and boundary-index compaction. It broadens an
already large NanoVDB MR and is unnecessary for S2.

### Partition-local connectivity mapping

A fixed cell-to-global-node map could avoid auxiliary NanoVDB volumes, but it
would require capture-safe sort/unique, inverse mappings, overflow reporting,
and broad topology/partition changes. Persistent auxiliary volumes fit the
existing Nanogrid lookup model with much less risk.

### Preserve the experimental patch unchanged

This is rejected because it relies on removed private fields and uses an
insufficient vertex-derived hierarchy bound for edge topology.
