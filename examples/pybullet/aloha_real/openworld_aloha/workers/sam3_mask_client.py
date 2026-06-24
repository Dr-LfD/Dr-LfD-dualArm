"""Lightweight SAM3 mask client for online use.

Talks to ``workers/sam3_worker.py`` over the same length-prefixed pickle
protocol defined in ``workers/ipc_utils.py``. Keeping the client next to the
worker avoids pulling in heavy training-only dependencies (cv2, h5py, networkx,
etc.) just to get per-frame object masks during real-time prediction.
"""

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

_WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

from ipc_utils import recv_msg, send_msg

# Fixed SAM3 detector postprocessor threshold (passed to ``sam3_worker`` init).
SAM3_DETECTION_THRESHOLD = 0.3


def _default_conda_bin() -> str:
    candidate = Path.home() / "miniforge3" / "bin" / "conda"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Conda executable not found at {candidate}. "
            "Pass conda_bin to Sam3MaskClient to override."
        )
    return str(candidate)


_shared_lock = threading.Lock()
_shared_clients: Dict[Tuple, "Sam3MaskClient"] = {}


def sam3_config_key(
    worker_script: str,
    conda_env: str,
    sam3_path: str,
    model_dir: Optional[str],
    checkpoint_path: Optional[str],
    conda_bin: Optional[str],
) -> Tuple:
    """Hashable normalized SAM3 worker configuration (one worker per distinct key)."""
    worker_script = os.path.abspath(os.path.expanduser(worker_script))
    sam3_path = os.path.abspath(os.path.expanduser(sam3_path))
    model_dir_n = (
        os.path.abspath(os.path.expanduser(model_dir)) if model_dir else None
    )
    checkpoint_n = (
        os.path.abspath(os.path.expanduser(checkpoint_path))
        if checkpoint_path
        else None
    )
    resolved_conda = (
        os.path.abspath(os.path.expanduser(conda_bin))
        if conda_bin
        else _default_conda_bin()
    )
    return (
        worker_script,
        str(conda_env),
        sam3_path,
        model_dir_n,
        checkpoint_n,
        resolved_conda,
    )


def get_shared_sam3_client(
    worker_script: str,
    conda_env: str,
    sam3_path: str,
    model_dir: Optional[str],
    checkpoint_path: Optional[str],
    conda_bin: Optional[str] = None,
) -> "Sam3MaskClient":
    """Return a process-wide shared ``Sam3MaskClient`` for this exact configuration."""
    key = sam3_config_key(
        worker_script,
        conda_env,
        sam3_path,
        model_dir,
        checkpoint_path,
        conda_bin,
    )
    with _shared_lock:
        existing = _shared_clients.get(key)
        proc = getattr(existing, "_proc", None) if existing is not None else None
        if existing is not None and proc is not None and proc.poll() is None:
            return existing
        if existing is not None:
            try:
                existing.close()
            except Exception:
                pass
            _shared_clients.pop(key, None)
        client = Sam3MaskClient(
            worker_script=key[0],
            conda_env=key[1],
            sam3_path=key[2],
            model_dir=key[3],
            checkpoint_path=key[4],
            conda_bin=key[5],
        )
        _shared_clients[key] = client
        return client


def close_shared_sam3_clients() -> None:
    """Terminate all shared SAM3 workers for this process."""
    with _shared_lock:
        for client in list(_shared_clients.values()):
            try:
                client.close()
            except Exception:
                pass
        _shared_clients.clear()


atexit.register(close_shared_sam3_clients)


class Sam3MaskClient:
    """Per-frame SAM3 mask client backed by ``workers/sam3_worker.py``."""

    def __init__(
        self,
        worker_script: str,
        conda_env: str,
        sam3_path: str,
        model_dir: Optional[str],
        checkpoint_path: Optional[str],
        conda_bin: Optional[str] = None,
    ):
        worker_script = os.path.abspath(worker_script)
        if not os.path.isfile(worker_script):
            raise FileNotFoundError(f"SAM3 worker script not found: {worker_script}")
        conda_bin = conda_bin or _default_conda_bin()

        self._proc = subprocess.Popen(
            [conda_bin, "run", "--no-capture-output", "-n", conda_env, "python", worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            bufsize=0,
        )
        self._tmpdir = tempfile.mkdtemp(prefix="sam3_mask_client_")

        try:
            send_msg(self._proc.stdin, {
                "cmd": "init",
                "sam3_path": sam3_path,
                "checkpoint_path": checkpoint_path,
                "model_dir": model_dir,
                "detection_threshold": SAM3_DETECTION_THRESHOLD,
            })
            response = recv_msg(self._proc.stdout)
            if response is None:
                raise RuntimeError("SAM3 worker terminated during initialization")
            if response.get("error"):
                raise RuntimeError(
                    f"SAM3 worker init failed: {response['error']}\n"
                    f"{response.get('traceback', '')}"
                )
        except Exception:
            self.close()
            raise

    def segment(self, rgb_image: np.ndarray, class_names: Sequence[str]) -> dict:
        if self._proc is None:
            raise RuntimeError("SAM3 worker is not running")
        # Write the frame as PNG; sam3_worker.py reads via PIL.
        import cv2
        image_path = os.path.join(self._tmpdir, "frame.png")
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(image_path, bgr):
            raise RuntimeError(f"Failed to write SAM3 frame to {image_path}")
        send_msg(self._proc.stdin, {
            "cmd": "seg",
            "image_path": image_path,
            "class_names": list(class_names),
        })
        response = recv_msg(self._proc.stdout)
        if response is None:
            raise RuntimeError("SAM3 worker exited unexpectedly")
        if response.get("error"):
            raise RuntimeError(
                f"SAM3 worker error: {response['error']}\n"
                f"{response.get('traceback', '')}"
            )
        return response.get("masks_by_class", {})

    def close(self):
        proc = self._proc
        if proc is not None:
            try:
                if proc.poll() is None:
                    send_msg(proc.stdin, {"cmd": "exit"})
                    recv_msg(proc.stdout)
            except Exception:
                pass
            for stream in (proc.stdin, proc.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
            self._proc = None

        tmpdir = getattr(self, "_tmpdir", None)
        if tmpdir and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
            self._tmpdir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def apply_object_mask(
    rgb_image: np.ndarray,
    masks_by_class: dict,
    class_name: str,
) -> np.ndarray:
    """Union all SAM3 masks for ``class_name`` and return ``rgb_image`` * mask.

    Returns a uint8 image with the same shape as ``rgb_image``. Pixels outside
    the mask are zeroed. If SAM3 returned no masks for ``class_name``, the
    output is an all-zero image (i.e. the predictor sees no object).
    """
    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"apply_object_mask expects HxWx3 RGB images, got shape {image.shape}."
        )
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    height, width = image.shape[:2]
    frame_mask = np.zeros((height, width), dtype=bool)
    for raw_mask in masks_by_class.get(class_name) or []:
        mask_arr = np.asarray(raw_mask, dtype=bool)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[0]
        if mask_arr.shape != (height, width):
            import cv2
            mask_arr = cv2.resize(
                mask_arr.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        frame_mask |= mask_arr

    return (image * frame_mask[..., None]).astype(np.uint8)
