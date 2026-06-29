# Rebuildable Nanogrid S2 Edge Topology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep a Nanogrid's S2 edge-index topology valid and fixed-capacity across rebuilds and CUDA graph replay.

**Architecture:** Reuse the MR's persistent auxiliary-volume pattern. Vertex and edge topology keep separate candidate-generation kernels but share one allocator that derives conservative capacities from public `Volume.get_rebuild_info()` metadata; vertex expansion is 8 and edge expansion is 12. Edge topology remains lazy, so Q1 users incur no new memory or rebuild work.

**Tech Stack:** Python 3.12, Warp FEM, NanoVDB index volumes, CUDA graph capture, `unittest`, `uv`, `uvx pre-commit`.

---

### Task 1: Add a failing S2 capture regression

**Files:**
- Modify: `warp/tests/fem/test_fem_geometry.py:371-510`

- [ ] **Step 1: Extend the volume-count kernel to observe edge topology**

Change the existing helper to accept the edge volume and record three active counts:

```python
@wp.kernel
def _nanogrid_volume_counts(
    cell_grid: wp.uint64,
    vertex_grid: wp.uint64,
    edge_grid: wp.uint64,
    counts: wp.array(dtype=wp.int32),
):
    counts[0] = wp.volume_voxel_count(cell_grid)
    counts[1] = wp.volume_voxel_count(vertex_grid)
    counts[2] = wp.volume_voxel_count(edge_grid)
```

Update both existing callers to allocate three counters and pass `geo.edge_grid.id`.

- [ ] **Step 2: Make the capture regression construct S2 topology**

Replace the Q1 setup in `test_nanogrid_rebuild_capture` with:

```python
space = fem.make_polynomial_space(
    geo,
    degree=2,
    element_basis=fem.ElementBasis.SERENDIPITY,
)
vertex_grid_id = geo.vertex_grid.id
edge_grid = geo.edge_grid  # Keep the pre-capture object alive intentionally.

test.assertEqual(space.topology._vertex_grid, vertex_grid_id)
test.assertEqual(space.topology._edge_grid, edge_grid.id)
```

Assert the initial counts are `[1, 8, 12]`. Capture the rebuild and count launch using `edge_grid.id`, then assert:

```python
test.assertIs(geo.edge_grid, edge_grid)
test.assertEqual(space.topology._edge_grid, edge_grid.id)
np.testing.assert_array_equal(counts.numpy(), np.array([3, 20, 32]))
```

- [ ] **Step 3: Run the focused test and verify RED**

Run:

```bash
uv run --extra dev warp/tests/fem/test_fem_geometry.py -k nanogrid_rebuild_capture -v
```

Expected: both CUDA-device cases fail because the retained edge volume still has 12 active edges after replay, or because `geo.edge_grid` is a different object. The failure must occur at the new S2 edge assertions, not during setup.

### Task 2: Add a failing edge-capacity boundary regression

**Files:**
- Modify: `warp/tests/fem/test_fem_geometry.py`

- [ ] **Step 1: Add a dense-leaf edge-capacity test**

Add a CUDA test that activates all cells in one `8 x 8 x 8` NanoVDB leaf:

```python
def test_nanogrid_rebuild_edge_capacity(test, device):
    cell_ijk = wp.array(
        np.stack(np.meshgrid(np.arange(8), np.arange(8), np.arange(8), indexing="ij"), axis=-1).reshape(-1, 3),
        dtype=wp.int32,
        device=device,
    )
    volume = wp.Volume.allocate_by_voxels(
        cell_ijk,
        voxel_size=1.0,
        device=device,
        rebuildable=True,
        max_active_voxels=cell_ijk.shape[0],
        max_leaf_nodes=1,
        max_lower_nodes=1,
        max_upper_nodes=1,
    )
    geo = fem.Nanogrid(volume, rebuildable=True)
    edge_grid = geo.edge_grid
    active = edge_grid.get_active_stats()
    capacity = edge_grid.get_rebuild_info()

    test.assertTrue(edge_grid.is_rebuildable)
    test.assertEqual(active.voxel_count, 3 * 8 * 9 * 9)
    test.assertEqual(active.leaf_node_count, 12)
    test.assertGreaterEqual(capacity.max_leaf_node_count, active.leaf_node_count)
```

Register it beside the existing Nanogrid rebuild tests on `cuda_devices`.

- [ ] **Step 2: Run the boundary test and verify RED**

Run:

```bash
uv run --extra dev warp/tests/fem/test_fem_geometry.py -k nanogrid_rebuild_edge_capacity -v
```

Expected: fail because the MR baseline creates a non-rebuildable active-sized edge volume.

### Task 3: Implement persistent rebuildable edge topology

**Files:**
- Modify: `warp/_src/fem/geometry/nanogrid.py:665-720`
- Modify: `warp/_src/fem/geometry/nanogrid.py:1048-1072`
- Modify: `warp/_src/fem/geometry/nanogrid.py:1593-1704`

- [ ] **Step 1: Retain lazy edge-rebuild state**

Initialize the state next to `_edge_grid`:

```python
self._edge_count = 0
self._edge_grid = None
self._edge_candidates = None
self._edge_candidate_mask = None
```

- [ ] **Step 2: Add masked edge candidate generation**

Add `_rebuildable_cell_edge_indices`, matching `_cell_edge_indices`' twelve coordinates and setting all twelve masks from:

```python
active = wp.where(cell < wp.volume_voxel_count(cell_grid), wp.int32(1), wp.int32(0))
for edge in range(12):
    edge_mask[cell * 12 + edge] = active
```

Add `_fill_rebuildable_edge_candidates()` to launch it over `cell_ijk.shape[0]`.

- [ ] **Step 3: Share auxiliary-volume capacity allocation**

Extract the common allocation portion of `_build_rebuildable_node_grid`:

```python
def _build_rebuildable_entity_grid(candidates, candidate_mask, grid: wp.Volume, hierarchy_expansion: int):
    entity_capacity = candidates.size
    rebuild_info = grid.get_rebuild_info()
    max_leaf_nodes = min(entity_capacity, rebuild_info.max_leaf_node_count * hierarchy_expansion)
    max_lower_nodes = min(max_leaf_nodes, rebuild_info.max_lower_node_count * hierarchy_expansion)
    max_upper_nodes = min(max_lower_nodes, rebuild_info.max_upper_node_count * hierarchy_expansion)
    return wp.Volume.allocate_by_voxels(
        candidates.flatten(),
        voxel_size=grid.get_voxel_size(),
        device=candidates.device,
        rebuildable=True,
        max_active_voxels=entity_capacity,
        max_leaf_nodes=max_leaf_nodes,
        max_lower_nodes=max_lower_nodes,
        max_upper_nodes=max_upper_nodes,
        point_mask=candidate_mask,
    )
```

Use it from `_build_rebuildable_node_grid(..., hierarchy_expansion=8)` and a new `_build_rebuildable_edge_grid(..., hierarchy_expansion=12)`.

- [ ] **Step 4: Build and refresh the persistent edge volume**

Make `_build_edge_grid` select the rebuildable path only when `self._rebuildable` is true. Set `_edge_count` from the edge volume's fixed `get_rebuild_info().max_voxel_count`.

In `_refresh_rebuildable_topology`, replace unconditional edge invalidation with:

```python
if self._edge_candidates is None:
    self._edge_grid = None
    self._edge_count = 0
else:
    _fill_rebuildable_edge_candidates(
        self._cell_grid,
        self._cell_ijk,
        self._edge_candidates,
        self._edge_candidate_mask,
    )
    self._edge_grid.rebuild(self._edge_candidates.flatten(), point_mask=self._edge_candidate_mask)
```

Keep face topology invalidation unchanged.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
uv run --extra dev warp/tests/fem/test_fem_geometry.py -k nanogrid_rebuild -v
```

Expected: four CUDA-device test cases pass with stable edge IDs, `12 -> 32` active edges, and the dense-leaf capacity bound.

### Task 4: Document the expanded Nanogrid behavior

**Files:**
- Modify: `CHANGELOG.md:30-34`
- Modify: `warp/_src/fem/geometry/nanogrid.py` docstrings as needed

- [ ] **Step 1: Update the existing unreleased changelog entry**

Change the NanoVDB entry to state that rebuildable Nanogrid topology refresh supports vertex- and edge-noded spaces.

- [ ] **Step 2: Clarify lazy topology lifecycle**

Document that auxiliary topology required by a FEM space is allocated when the space is constructed and then refreshed in place during rebuilds.

- [ ] **Step 3: Format and lint changed files**

Run:

```bash
uvx pre-commit run --files CHANGELOG.md warp/_src/fem/geometry/nanogrid.py warp/tests/fem/test_fem_geometry.py
```

Expected: all hooks pass and no files remain reformatted but unstaged.

### Task 5: Verify Warp behavior comprehensively

**Files:**
- Test: `warp/tests/fem/test_fem_geometry.py`

- [ ] **Step 1: Run the full FEM geometry module**

```bash
uv run --extra dev warp/tests/fem/test_fem_geometry.py -v
```

Expected: all CPU and CUDA cases pass on both selected GPUs.

- [ ] **Step 2: Run the MR-focused rebuild tests**

```bash
uv run --extra dev warp/tests/fem/test_fem_geometry.py -k nanogrid_rebuild -v
```

Expected: all rebuild, edge-capacity, and capture cases pass.

- [ ] **Step 3: Inspect the final diff**

```bash
git diff --check
git diff --stat internal/mr-2533...HEAD
```

Expected: only the design/plan documents, changelog, Nanogrid implementation, and FEM geometry tests differ from the MR head.

- [ ] **Step 4: Commit the implementation**

```bash
git add CHANGELOG.md warp/_src/fem/geometry/nanogrid.py warp/tests/fem/test_fem_geometry.py
git commit --signoff -m "Support rebuildable Nanogrid edge topology"
```

### Task 6: Validate Newton MPM S2 end to end

**Files:**
- Read-only consumer: `/home/maximiliank/Work/newton-coupled`
- Read-only demo: `/home/maximiliank/Work/IsaacLab-coupling/scripts/demos/mpm/lava_flow.py`

- [ ] **Step 1: Run the eager control**

Launch the lava demo through the IsaacLab environment with this Warp worktree first on `PYTHONPATH`, sparse grid, S2 collider, graph disabled, and at least 45 steps.

Expected: completion without CUDA errors and a `[PERF]` line reporting `graph=off`.

- [ ] **Step 2: Run CUDA graph capture and replay**

Launch the same demo with graph enabled for at least 70 steps.

Expected: a `CUDA graph took` capture record, completion without fallback or CUDA/capacity errors, and a `[PERF]` line reporting `graph=on`.

- [ ] **Step 3: Record exact commands and evidence**

Preserve the command lines, selected device, Warp commit, test counts, capture evidence, and performance lines in the final handoff.

### Task 7: Independent review and completion verification

**Files:**
- Review all changes relative to `internal/mr-2533`

- [ ] **Step 1: Request spec-compliance review**

Have an independent reviewer compare the implementation and tests against the approved design, including the edge-only scope and factor-twelve capacity argument.

- [ ] **Step 2: Request code-quality review**

Have a separate reviewer inspect correctness, graph-capture lifecycle, capacity math, multi-environment behavior, and test quality.

- [ ] **Step 3: Re-run verification after review fixes**

Repeat focused tests, full FEM geometry tests, pre-commit, and the Newton graph-on smoke after any review-driven edit.
