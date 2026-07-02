"""Numpy-only reachability queries against the precomputed ALOHA coverage map.

This module replaces the legacy ad-hoc reachability heuristics (radial/AABB in the
MuJoCo path, left/right x-split in the real path). It loads only the compact coverage
artifact produced offline by ``reachabilitymap/tools/export_aloha_coverage.py`` and
reproduces the map's voxel indexing and the paper's position-only ``IsReachable``
(coverage fraction >= threshold). It imports neither ``reach_map`` nor torch.

The artifact stores, per workspace voxel, the fraction of orientations a single vx300s
arm can reach in its own base frame. Both ALOHA arms are identical vx300s units (the
right arm is the same arm rotated 180 deg about z), so one map serves both arms once a
world point is expressed in the queried arm's base frame via ``inv(*_BASE_POSE)``.
"""

import threading

import numpy as np

from examples.pybullet.aloha_real.scripts.aloha_tamp_constants import (
    LEFT_BASE_POSE,
    RIGHT_BASE_POSE,
)


class AlohaReachability:
    """Position-only reachability lookup over the compact coverage voxel map."""

    def __init__(self, artifact_path, coverage_threshold):
        self.artifact_path = str(artifact_path)
        self.coverage_threshold = float(coverage_threshold)

        # np.load raises FileNotFoundError if the artifact is missing; this is the
        # intended developer-visible failure (no silent fallback to a heuristic).
        with np.load(self.artifact_path, allow_pickle=False) as data:
            self.coverage = np.ascontiguousarray(data["coverage"], dtype=np.float32)
            self.cell_size = float(data["cell_size"])
            self.limits_per_dim = np.asarray(data["limits_per_dim"], dtype=np.float64)
            self.ndim = int(data["ndim"])
            self.approach_axis = str(data["approach_axis"])
            self.distance_threshold = float(data["distance_threshold"])
            saved_num_cells_per_dim = np.asarray(data["num_cells_per_dim"], dtype=np.int64)

        if self.ndim != 3 or self.limits_per_dim.shape != (3, 2):
            raise ValueError(
                f"unexpected map geometry: ndim={self.ndim}, "
                f"limits_per_dim shape={self.limits_per_dim.shape}"
            )

        # Reproduce reach_map.base_grid.HyperGridBase exactly: ceil for the per-dim
        # cell counts, then row-major strides (last stride == 1).
        lo = self.limits_per_dim[:, 0]
        hi = self.limits_per_dim[:, 1]
        self.num_cells_per_dim = np.ceil((hi - lo) / self.cell_size).astype(np.int64)
        # Cross-check the recomputed dims against the builder's: a mismatch means the
        # artifact's geometry no longer matches this indexing (e.g. float-rounding skew).
        if not np.array_equal(self.num_cells_per_dim, saved_num_cells_per_dim):
            raise ValueError(
                f"num_cells_per_dim mismatch: recomputed {self.num_cells_per_dim.tolist()} "
                f"vs artifact {saved_num_cells_per_dim.tolist()}"
            )
        self.strides = np.ones(self.ndim, dtype=np.int64)
        for i in range(self.ndim - 1):
            self.strides[i] = int(np.prod(self.num_cells_per_dim[i + 1:]))

        expected = int(np.prod(self.num_cells_per_dim))
        if self.coverage.shape != (expected,):
            raise ValueError(
                f"coverage length {self.coverage.shape} != prod(num_cells_per_dim)={expected}"
            )

        self._world_to_base = {
            "left": np.linalg.inv(np.asarray(LEFT_BASE_POSE, dtype=np.float64)),
            "right": np.linalg.inv(np.asarray(RIGHT_BASE_POSE, dtype=np.float64)),
        }

    def _cell_id(self, pos_xyz):
        """Voxel id for a point in the arm base frame, or None if outside the map.

        Unlike the upstream grid (which clamps out-of-range positions to the edge
        cell), points outside ``limits_per_dim`` are rejected: a target beyond the
        built workspace is genuinely unreachable, not edge-reachable.
        """
        pos = np.asarray(pos_xyz, dtype=np.float64).reshape(-1)
        if pos.shape != (3,):
            raise ValueError(f"expected a 3-vector position, got shape {pos.shape}")
        lo = self.limits_per_dim[:, 0]
        hi = self.limits_per_dim[:, 1]
        if np.any(pos < lo) or np.any(pos >= hi):
            return None
        # torch.Tensor.int() truncates toward zero; positions here are >= lo so the
        # truncation equals floor. astype(int64) matches that truncation.
        idx = ((pos - lo) / self.cell_size).astype(np.int64)
        return int(np.dot(idx, self.strides))

    def reachable_base(self, pos_base_xyz, threshold=None):
        """True if the voxel containing a base-frame point clears the coverage threshold.
        ``threshold`` overrides the oracle default (for per-object thresholds)."""
        cell_id = self._cell_id(pos_base_xyz)
        if cell_id is None:
            return False
        thr = self.coverage_threshold if threshold is None else float(threshold)
        return bool(self.coverage[cell_id] >= thr)

    def _world_to_base_point(self, side, world_point_xyz):
        """Express a world point in ``side``'s arm base frame (3-vector)."""
        if side not in self._world_to_base:
            raise ValueError(f"unknown ALOHA side {side!r}; expected 'left' or 'right'")
        point = np.ones(4, dtype=np.float64)
        point[:3] = np.asarray(world_point_xyz, dtype=np.float64).reshape(-1)
        return (self._world_to_base[side] @ point)[:3]

    def reachable(self, side, world_point_xyz, threshold=None):
        """Position-only IsReachable for ``side`` ('left'/'right') at a world point.
        ``threshold`` overrides the oracle default (per-object thresholds)."""
        return self.reachable_base(
            self._world_to_base_point(side, world_point_xyz), threshold=threshold)

_CACHE = {}
_LOCK = threading.Lock()


def get_reachability(artifact_path, coverage_threshold):
    """Process-wide singleton per (artifact, threshold); loads the map once."""
    key = (str(artifact_path), float(coverage_threshold))
    with _LOCK:
        oracle = _CACHE.get(key)
        if oracle is None:
            oracle = AlohaReachability(*key)
            _CACHE[key] = oracle
        return oracle
