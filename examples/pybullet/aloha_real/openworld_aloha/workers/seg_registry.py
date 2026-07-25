import os
import sys

_VALID_SEG_BACKENDS = frozenset({"sam3", "grounded_sam"})


def canonicalize_seg_backend(raw):
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise ValueError("seg_branch is required.")
    value = str(raw).strip()
    if value == "sam":
        raise ValueError("Use 'grounded_sam', not legacy 'sam'.")
    if value not in _VALID_SEG_BACKENDS:
        raise ValueError(f"Invalid seg_branch: {value!r}.")
    return value


def init_seg_branch_for_canonical(backend):
    # estimation_policy's legacy branch id: grounded_sam reuses the old "sam" class name.
    if backend == "grounded_sam":
        return "sam"
    if backend == "sam3":
        return "sam3"
    raise ValueError(f"Invalid seg backend: {backend!r}.")


class SegReuseRegistry:
    _backend = None
    _sam3_client = None
    _grounded_sam_segmenter = None

    @classmethod
    def init(cls, backend, para):
        cls._backend = backend
        cls._sam3_client = None
        cls._grounded_sam_segmenter = None
        if backend == "sam3":
            cls._sam3_client = cls._create_sam3_client(para)
        elif backend == "grounded_sam":
            cls._grounded_sam_segmenter = cls._create_grounded_sam_segmenter(para)

    @classmethod
    def backend(cls):
        if cls._backend is None:
            raise RuntimeError("SegReuseRegistry.init was not called.")
        return cls._backend

    @classmethod
    def get_sam3_client(cls):
        if cls.backend() != "sam3":
            raise RuntimeError(f"SAM3 inactive: {cls.backend()!r}.")
        if cls._sam3_client is None:
            raise RuntimeError("SAM3 client was not initialized.")
        return cls._sam3_client

    @classmethod
    def get_grounded_sam_segmenter(cls):
        if cls.backend() != "grounded_sam":
            raise RuntimeError(f"Grounded SAM inactive: {cls.backend()!r}.")
        if cls._grounded_sam_segmenter is None:
            raise RuntimeError("GroundedSamSegmenter was not initialized.")
        return cls._grounded_sam_segmenter

    @staticmethod
    def _create_sam3_client(para):
        worker_path = para.get("sam3_worker_path")
        sam3_path = para.get("sam3_path")
        if not worker_path or not sam3_path:
            raise ValueError("SAM3 requires worker and repository paths.")
        from examples.pybullet.aloha_real.openworld_aloha.workers.sam3_mask_client import (
            get_shared_sam3_client,
        )

        return get_shared_sam3_client(
            worker_script=os.path.abspath(os.path.expanduser(worker_path)),
            conda_env=para.get("sam3_conda_env", "sam3"),
            sam3_path=os.path.abspath(os.path.expanduser(sam3_path)),
            model_dir=para.get("sam3_model_dir"),
            checkpoint_path=para.get("sam3_checkpoint"),
            conda_bin=para.get("sam3_conda_bin"),
        )

    @staticmethod
    def _create_grounded_sam_segmenter(para):
        sam_path = para.get("sam_path")
        if not sam_path:
            raise ValueError("Grounded SAM requires a repository path.")
        repo_root = os.path.abspath(os.path.expanduser(sam_path))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        # grounded_sam's provider was dropped from contact_detector; this raises if actually selected.
        from contact_pred.scripts.common.grounded_sam_segmenter import GroundedSamSegmenter

        return GroundedSamSegmenter({"repo_root": repo_root})
