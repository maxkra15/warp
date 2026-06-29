# Rebuildable Nanogrid Edge Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make rebuildable Nanogrid S2 capture fail safely for unprepared topology and report auxiliary topology rebuild overflow through the existing status value.

**Architecture:** Retain lazy edge allocation and lock the prepared topology set when a refresh executes during CUDA graph capture. Rebuild node and edge volumes into persistent scratch statuses, then use one capture-safe kernel to merge their raw `Volume.REBUILD_*` bits into the caller's status; reject unsupported face-noded topology before it can become stale.

**Tech Stack:** Python 3.12, Warp FEM, NanoVDB index volumes, CUDA graph capture, `unittest`, `uv`, and `uvx pre-commit`.

---

### Task 1: Pin lifecycle and face failures

**Files:**
- Modify: `warp/tests/fem/test_fem_geometry.py:468-570`
- Modify: `warp/_src/fem/geometry/nanogrid.py:700-730,1030-1097`
- Modify: `warp/_src/fem/space/nanogrid_function_space.py:28-50`

- [ ] **Step 1: Write the failing lifecycle tests**

Add a CUDA test that captures a node-only rebuild and then requests S2:

```python
def test_nanogrid_rebuild_capture_topology_lock(test, device):
    points = wp.array([[0, 0, 0]], dtype=wp.int32, device=device)
    volume = wp.Volume.allocate_by_voxels(
        points,
        voxel_size=1.0,
        device=device,
        rebuildable=True,
        max_active_voxels=1,
        max_leaf_nodes=1,
        max_lower_nodes=1,
        max_upper_nodes=1,
    )
    geo = fem.Nanogrid(volume, rebuildable=True)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False):
        geo.rebuild(points)

    with test.assertRaisesRegex(RuntimeError, "before CUDA graph capture"):
        fem.make_polynomial_space(geo, degree=2, element_basis=fem.ElementBasis.SERENDIPITY)
```

In the same test, construct a second rebuildable Nanogrid, call `side_count()` to materialize dynamic face state, and assert that capturing `geo_with_faces.rebuild(points)` raises `RuntimeError` containing `"face topology"`.

Add a device-independent face-noded-space test:

```python
def test_nanogrid_rebuild_face_topology(test, device):
    points = wp.array([[0, 0, 0]], dtype=wp.int32, device=device)
    volume = wp.Volume.allocate_by_voxels(points, 1.0, device=device, rebuildable=True)
    geo = fem.Nanogrid(volume, rebuildable=True)

    with test.assertRaisesRegex(NotImplementedError, "Face-noded spaces"):
        fem.make_polynomial_space(geo, degree=2)
```

Register the capture-lock test on `cuda_devices` and the face-topology test on `devices`.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run --extra dev warp/tests/fem/test_fem_geometry.py -k topology_lock -k face_topology -v
```

Expected: the late S2 request and face-noded space do not raise the requested errors on the current branch.

- [ ] **Step 3: Implement the private capture lock**

Initialize one private boolean in `Nanogrid.__init__`:

```python
self._topology_capture_locked = False
```

At the start of `_refresh_rebuildable_topology()`, enforce the captured topology set:

```python
if self._cell_grid.device.is_capturing:
    if self._face_grid is not None:
        raise RuntimeError("Rebuildable Nanogrid face topology is not supported during CUDA graph capture")
    self._topology_capture_locked = True
```

Guard lazy allocation without adding public lifecycle methods:

```python
def _ensure_face_grid(self):
    if self._rebuildable and (self._topology_capture_locked or self._cell_grid.device.is_capturing):
        raise RuntimeError(
            "Cannot materialize rebuildable Nanogrid face topology after CUDA graph capture; "
            "persistent face topology is unsupported"
        )
    super()._ensure_face_grid()

def _ensure_edge_grid(self):
    if self._edge_grid is None:
        if self._rebuildable and (self._topology_capture_locked or self._cell_grid.device.is_capturing):
            raise RuntimeError(
                "Materialize rebuildable Nanogrid edge topology before CUDA graph capture by constructing its FEM space"
            )
        self._build_edge_grid()
```

- [ ] **Step 4: Reject face-noded rebuildable spaces before allocation**

Compute the shape requirements at the beginning of `NanogridSpaceTopology.__init__`, before `super().__init__`, and add:

```python
need_edge_indices = shape.EDGE_NODE_COUNT > 0
need_face_indices = shape.FACE_NODE_COUNT > 0
if isinstance(grid, Nanogrid) and grid._rebuildable and need_face_indices:
    raise NotImplementedError("Face-noded spaces are not supported for rebuildable Nanogrids")
```

Keep AdaptiveNanogrid and non-rebuildable Nanogrid behavior unchanged.

- [ ] **Step 5: Run the tests and verify GREEN**

Run the command from Step 2. Expected: every selected CPU/CUDA face test and CUDA lifecycle test passes.

- [ ] **Step 6: Commit the lifecycle fix**

```bash
git add warp/_src/fem/geometry/nanogrid.py warp/_src/fem/space/nanogrid_function_space.py warp/tests/fem/test_fem_geometry.py
git commit --signoff -m "Guard captured Nanogrid topology"
```

### Task 2: Propagate auxiliary rebuild status

**Files:**
- Modify: `warp/tests/fem/test_fem_geometry.py:371-570`
- Modify: `warp/_src/fem/geometry/nanogrid.py:700-805,1072-1097,1605-1610`

- [ ] **Step 1: Write the failing auxiliary-overflow test**

Add a test on `devices` that materializes edge candidates, substitutes a deliberately undersized rebuildable edge volume, and checks eager status propagation:

```python
def test_nanogrid_rebuild_auxiliary_status(test, device):
    points = wp.array([[0, 0, 0]], dtype=wp.int32, device=device)
    status = wp.zeros(1, dtype=wp.uint32, device=device)
    volume = wp.Volume.allocate_by_voxels(
        points,
        voxel_size=1.0,
        device=device,
        rebuildable=True,
        max_active_voxels=1,
        max_leaf_nodes=1,
        max_lower_nodes=1,
        max_upper_nodes=1,
    )
    geo = fem.Nanogrid(volume, rebuildable=True)
    _ = geo.edge_grid
    geo._edge_grid = wp.Volume.allocate_by_voxels(
        points,
        voxel_size=1.0,
        device=device,
        rebuildable=True,
        max_active_voxels=1,
        max_leaf_nodes=1,
        max_lower_nodes=1,
        max_upper_nodes=1,
    )
    geo._edge_count = 1

    geo.rebuild(points, status=status)
    test.assertTrue(int(status.numpy()[0]) & wp.Volume.REBUILD_VOXEL_CAPACITY_EXCEEDED)
```

For CUDA, clear `status`, capture `geo.rebuild(points, status=status)`, replay once, and assert the same raw flag. The eager call intentionally compiles the aggregation kernel before capture.

- [ ] **Step 2: Verify the status test fails**

Run:

```bash
uv run --extra dev warp/tests/fem/test_fem_geometry.py -k nanogrid_rebuild_auxiliary_status -v
```

Expected: `status` remains `Volume.REBUILD_SUCCESS` because the current node and edge rebuilds receive no status.

- [ ] **Step 3: Add one capture-safe aggregation kernel**

Add this kernel beside the topology-building kernels:

```python
@wp.kernel
def _aggregate_nanogrid_rebuild_status(
    status: wp.array(dtype=wp.uint32),
    node_status: wp.array(dtype=wp.uint32),
    edge_status: wp.array(dtype=wp.uint32),
    include_edge: int,
    preserve_status: int,
):
    value = node_status[0]
    if include_edge != 0:
        value = value | edge_status[0]
    if preserve_status != 0:
        value = value | status[0]
    status[0] = value
```

- [ ] **Step 4: Allocate only the required scratch state**

For rebuildable Nanogrids, allocate two persistent one-element arrays; use `None` for fixed Nanogrids:

```python
self._node_rebuild_status = wp.empty(1, dtype=wp.uint32, device=device) if rebuildable else None
self._edge_rebuild_status = wp.empty(1, dtype=wp.uint32, device=device) if rebuildable else None
```

- [ ] **Step 5: Thread status through topology refresh**

Change `rebuild_topology_from_cells` to accept `status: wp.array | None = None`. Change `_refresh_rebuildable_topology` to accept `status` and `preserve_status`.

When `status` is present, rebuild the node volume into `_node_rebuild_status` if preserving the cell status, otherwise rebuild it directly into the caller's status. Rebuild a prepared edge volume into `_edge_rebuild_status`. Finish with exactly one aggregation launch:

```python
node_status = self._node_rebuild_status if preserve_status and status is not None else status
self._node_grid.rebuild(
    self._node_candidates.flatten(),
    status=node_status,
    point_mask=self._node_candidate_mask,
)

include_edge = self._edge_candidates is not None
if include_edge:
    self._edge_grid.rebuild(
        self._edge_candidates.flatten(),
        status=self._edge_rebuild_status if status is not None else None,
        point_mask=self._edge_candidate_mask,
    )

if status is not None:
    wp.launch(
        _aggregate_nanogrid_rebuild_status,
        dim=1,
        inputs=[status, node_status, self._edge_rebuild_status, int(include_edge), int(preserve_status)],
        device=self._cell_grid.device,
    )
```

`Nanogrid.rebuild()` calls `_refresh_rebuildable_topology(status=status, preserve_status=True)` after the cell rebuild. `rebuild_topology_from_cells(status)` calls it with `preserve_status=False`. If `status` is `None`, retain the existing unchecked `Volume.rebuild()` convention and do not launch the aggregation kernel.

- [ ] **Step 6: Run the focused tests and verify GREEN**

Run:

```bash
uv run --extra dev warp/tests/fem/test_fem_geometry.py -k nanogrid_rebuild_auxiliary_status -k nanogrid_rebuild_capture -v
```

Expected: eager and captured auxiliary overflow expose the raw voxel-capacity bit, and the existing S2 capture regression still passes.

- [ ] **Step 7: Commit status propagation**

```bash
git add warp/_src/fem/geometry/nanogrid.py warp/tests/fem/test_fem_geometry.py
git commit --signoff -m "Report Nanogrid topology overflow"
```

### Task 3: Strengthen the capacity and count contract

**Files:**
- Modify: `warp/tests/fem/test_fem_geometry.py:468-570`

- [ ] **Step 1: Add the full-hierarchy capacity regression**

Create a rebuildable one-cell grid with voxel, leaf, lower, and upper capacities of one. Retain its edge grid, rebuild the cell from `(0, 0, 0)` to `(4095, 4095, 4095)`, and assert:

```python
test.assertEqual(int(status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)
test.assertIs(geo.edge_grid, edge_grid)
test.assertEqual(edge_grid.get_active_stats().voxel_count, 12)
test.assertEqual(edge_grid.get_active_stats().leaf_node_count, 12)
test.assertEqual(edge_grid.get_active_stats().lower_node_count, 12)
test.assertEqual(edge_grid.get_active_stats().upper_node_count, 12)
test.assertEqual(edge_grid.get_rebuild_info().max_voxel_count, 12)
test.assertEqual(edge_grid.get_rebuild_info().max_leaf_node_count, 12)
test.assertEqual(edge_grid.get_rebuild_info().max_lower_node_count, 12)
test.assertEqual(edge_grid.get_rebuild_info().max_upper_node_count, 12)
```

Register this test on `devices` so the bound is checked on CPU and CUDA.

- [ ] **Step 2: Pin reserved and active counts in the existing capture test**

Before capture, assert:

```python
test.assertEqual(geo.edge_count(), 48)
test.assertEqual(edge_grid.get_rebuild_info().max_voxel_count, 48)
test.assertEqual(edge_grid.get_active_stats().voxel_count, 12)
test.assertEqual(space.node_count(), 80)
```

After replay, assert active edges are `32` while `edge_count()`, maximum edge capacity, and `space.node_count()` remain `48`, `48`, and `80`.

- [ ] **Step 3: Run the capacity tests**

Run:

```bash
uv run --extra dev warp/tests/fem/test_fem_geometry.py -k edge_capacity -k rebuild_capture -v
```

Expected: all selected CPU/CUDA cases pass; the new adversarial case reports twelve at every hierarchy level.

- [ ] **Step 4: Commit the contract tests**

```bash
git add warp/tests/fem/test_fem_geometry.py
git commit --signoff -m "Test Nanogrid edge capacity bounds"
```

### Task 4: Keep user-facing documentation minimal and accurate

**Files:**
- Modify: `warp/_src/fem/geometry/nanogrid.py:583-621,673-687,736-803`
- Modify: `CHANGELOG.md:40-43`

- [ ] **Step 1: Tighten existing docstrings**

Update only the existing rebuildable/status descriptions to state that required edge-noded spaces must be constructed before capture, later topology materialization raises, face-noded spaces remain unsupported, and the returned status is the union of cell and prepared auxiliary `Volume.REBUILD_*` flags.

- [ ] **Step 2: Amend the existing unreleased entry**

Add one sentence to the existing rebuildable NanoVDB bullet: `Nanogrid rebuild status includes prepared auxiliary topology, and captured refreshes reject later topology materialization.` Do not add a second changelog bullet.

- [ ] **Step 3: Run formatting and lint checks**

```bash
uvx pre-commit run --files CHANGELOG.md warp/_src/fem/geometry/nanogrid.py warp/_src/fem/space/nanogrid_function_space.py warp/tests/fem/test_fem_geometry.py
```

Expected: every hook passes. If a hook reformats a file, stage that exact formatter output and rerun the same command until all hooks pass.

- [ ] **Step 4: Commit the documentation adjustment**

```bash
git add CHANGELOG.md warp/_src/fem/geometry/nanogrid.py warp/_src/fem/space/nanogrid_function_space.py warp/tests/fem/test_fem_geometry.py
git commit --signoff -m "Clarify captured Nanogrid topology"
```

### Task 5: Verify Warp and the Newton consumer

**Files:**
- Verify: `warp/tests/fem/test_fem_geometry.py`
- Verify read-only: `/home/maximiliank/.config/superpowers/worktrees/newton/sparse-rebuildable/newton/tests/test_implicit_mpm_rebuildable_sparse.py`

- [ ] **Step 1: Run all focused Warp regressions**

```bash
uv run --extra dev warp/tests/fem/test_fem_geometry.py -k nanogrid_rebuild -v
```

Expected: rebuild, auxiliary status, lifecycle, face rejection, hierarchy capacity, and CUDA capture cases all pass.

- [ ] **Step 2: Run the complete FEM geometry module**

```bash
uv run --extra dev warp/tests/fem/test_fem_geometry.py -v
```

Expected: all CPU and selected CUDA tests pass with only platform/feature skips already present on the branch.

- [ ] **Step 3: Run Newton's isolated sparse-rebuildable regression against this Warp worktree**

From `/home/maximiliank/.config/superpowers/worktrees/newton/sparse-rebuildable`, run:

```bash
PYTHONPATH=/home/maximiliank/.config/superpowers/worktrees/warp-main-multiworld-reference/max-s2-rebuildable-edge-grid uv run --extra dev -m newton.tests -k implicit_mpm_rebuildable_sparse
```

Expected: all supported CPU/CUDA sparse-rebuildable tests pass; feature skips remain valid on devices without capture support.

- [ ] **Step 4: Inspect branch integrity**

```bash
git diff --check
git status --short --branch
git log --oneline --decorate -12
```

Expected: no unstaged implementation changes, a clean feature branch ahead of `fork/max/s2-rebuildable-edge-grid`, and no Newton file modifications.

- [ ] **Step 5: Request independent review and rerun affected checks**

Have one reviewer compare the implementation with the approved design and another inspect capture ordering, status aliasing, face rejection, and hierarchy tests. Apply only concrete in-scope fixes, then rerun Steps 1-4.

- [ ] **Step 6: Push without opening a PR**

```bash
git push fork max/s2-rebuildable-edge-grid
```

Expected: the fork branch advances and no pull request is created.
