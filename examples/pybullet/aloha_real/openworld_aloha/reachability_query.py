"""Numpy-only reachability queries against the precomputed ALOHA coverage map.


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

        # Load the precomputed coverage map and raise if its artifact is missing.
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

        # Rebuild the artifact's row-major grid dimensions and strides.
        lo = self.limits_per_dim[:, 0]
        hi = self.limits_per_dim[:, 1]
        self.num_cells_per_dim = np.ceil((hi - lo) / self.cell_size).astype(np.int64)
        # Reject artifacts whose stored dimensions disagree with the grid geometry.
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
        """Return the base-frame voxel ID for ``pos_xyz``, or None when it is outside the map."""
        pos = np.asarray(pos_xyz, dtype=np.float64).reshape(-1)
        if pos.shape != (3,):
            raise ValueError(f"expected a 3-vector position, got shape {pos.shape}")
        lo = self.limits_per_dim[:, 0]
        hi = self.limits_per_dim[:, 1]
        if np.any(pos < lo) or np.any(pos >= hi):
            return None
        # Return the row-major cell index for an in-bounds base-frame position.
        idx = ((pos - lo) / self.cell_size).astype(np.int64)
        return int(np.dot(idx, self.strides))

    def reachable_base(self, pos_base_xyz, threshold=None):
        """Return whether ``pos_base_xyz`` meets the optional coverage threshold."""
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
        """Return whether ``side`` can reach ``world_point_xyz`` at the optional threshold."""
        return self.reachable_base(
            self._world_to_base_point(side, world_point_xyz), threshold=threshold)

_CACHE = {}
_LOCK = threading.Lock()


def get_reachability(artifact_path, coverage_threshold):
    """Return the cached reachability map for an artifact and coverage threshold."""
    key = (str(artifact_path), float(coverage_threshold))
    with _LOCK:
        oracle = _CACHE.get(key)
        if oracle is None:
            oracle = AlohaReachability(*key)
            _CACHE[key] = oracle
        return oracle
