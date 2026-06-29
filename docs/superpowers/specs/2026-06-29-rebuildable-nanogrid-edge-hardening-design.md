# Rebuildable Nanogrid Edge Hardening Design

## Goal

Close the remaining correctness gaps in rebuildable Nanogrid S2 topology without
adding general face support or a new public lifecycle API.

## Design

Keep the existing lazy edge allocation. Constructing an edge-noded FEM space is
the preparation step: it materializes the persistent edge grid before capture.
When a topology refresh runs during CUDA graph capture, the Nanogrid records that
its current topology set is locked. A later attempt to materialize an edge grid
that was absent from the captured refresh fails with a clear error instead of
creating topology that the captured graph will never update. Face state also
cannot be materialized after this lock.

Face topology remains unsupported for rebuildable Nanogrids. Face-noded FEM
spaces fail at construction, and a captured rebuild fails if dynamic geometry
face state was already materialized. This prevents cached or newly allocated
face data from silently becoming stale.

Auxiliary rebuild failures must reach the status supplied to
`Nanogrid.rebuild()` or `rebuild_topology_from_cells()`. Node and edge rebuilds
use persistent one-element scratch status arrays allocated with the rebuildable
Nanogrid. One capture-safe kernel ORs their raw `Volume.REBUILD_*` flags into the
caller's status after the auxiliary rebuilds. No Nanogrid-specific status flags
or downstream Newton changes are introduced.

## Capacity and Count Contract

The existing entity allocator remains unchanged: vertex hierarchy capacities
use the proven factor eight and edge hierarchy capacities use the proven factor
twelve. Add an adversarial cell placement that realizes the twelve-edge bound at
the voxel, leaf, lower, and upper levels.

`edge_count()` continues to report reserved edge capacity so FEM category
offsets and field sizes remain fixed. Active edges remain available through
`edge_grid.get_active_stats()`. Rebuilds preserve the Volume object and ID, but
do not promise stable coordinate-to-index assignments inside that capacity.

## Tests

- Force an auxiliary edge overflow and verify its raw capacity flag reaches the
  Nanogrid status, including through CUDA graph replay.
- Capture a rebuild without edge topology, then verify that later edge
  materialization fails rather than creating stale topology.
- Verify that face-noded spaces on rebuildable Nanogrids fail immediately while
  S2 remains supported.
- Exercise the twelve-edge hierarchy bound at every NanoVDB level.
- Pin reserved versus active edge counts and stable S2 space size across rebuild.
- Retain the existing eager/captured topology identity and parity regressions.

## Exclusions

Do not add public prepare/freeze/query methods, a generic auxiliary-entity class,
persistent face metadata, coordinate-key redesign, or Newton changes. The
existing axis-tag coordinate limitation and full face topology are independent
follow-up work.
