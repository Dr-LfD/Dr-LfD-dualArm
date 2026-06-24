"""SAM3 worker process — runs inside the `sam3` conda env.

Uses the image model API (build_sam3_image_model) from detect_segment_video.py
instead of the video predictor API, to avoid stdout pollution that corrupts the
binary IPC protocol.
"""
import argparse
import io
import os
import sys
import traceback

import numpy as np
import torch

# Make ipc_utils importable when this script is launched directly by conda run.
_WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

from ipc_utils import recv_msg, send_msg


class Sam3Worker:
    def __init__(self):
        self.model = None
        self.transform = None
        self.postprocessor = None
        self._run_detection = None

    def _init(self, sam3_path, checkpoint_path, model_dir, detection_threshold):
        if not os.path.isdir(sam3_path):
            raise FileNotFoundError(f"SAM3 repo path does not exist: {sam3_path}")

        scripts_path = os.path.join(sam3_path, "scripts")
        sys.path.insert(0, sam3_path)
        sys.path.insert(0, scripts_path)

        from detect_segment_video import (
            build_image_model,
            build_postprocessor,
            build_transform,
            run_detection,
        )

        self.model = build_image_model(checkpoint_path, model_dir, compile=False)
        self.transform = build_transform()
        self.postprocessor = build_postprocessor(detection_threshold)
        self._run_detection = run_detection

    def _seg(self, image_path, class_names):
        if self.model is None:
            raise RuntimeError("SAM3 worker is not initialized — send INIT first")

        from PIL import Image as PILImage

        pil_image = PILImage.open(image_path).convert("RGB")

        with torch.inference_mode():
            detections = self._run_detection(
                self.model, self.transform, self.postprocessor,
                pil_image, class_names,
            )

        masks_by_class = {}
        for det in detections:
            name = det["class_name"]
            mask = np.asarray(det["mask"], dtype=bool)
            if mask.ndim == 3:
                mask = mask[0]
            if np.any(mask):
                masks_by_class.setdefault(name, []).append(mask)
        return masks_by_class

    def handle(self, request):
        cmd = request.get("cmd")
        if cmd == "init":
            self._init(
                sam3_path=request["sam3_path"],
                checkpoint_path=request.get("checkpoint_path"),
                model_dir=request.get("model_dir"),
                detection_threshold=request.get("detection_threshold", 0.5),
            )
            return {"ok": True}
        if cmd == "seg":
            masks_by_class = self._seg(
                image_path=request["image_path"],
                class_names=request["class_names"],
            )
            return {"masks_by_class": masks_by_class}
        if cmd == "exit":
            return {"ok": True, "exit": True}
        raise ValueError(f"Unknown command: {cmd!r}")


def _run_self_test():
    payload = {"hello": "world", "arr": list(range(10))}
    buf = io.BytesIO()
    send_msg(buf, payload)
    buf.seek(0)
    decoded = recv_msg(buf)
    assert decoded == payload, f"IPC roundtrip mismatch: {decoded}"
    print("sam3_worker self-test passed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _run_self_test()
        return 0

    # Reserve the raw stdout binary stream for IPC only, then redirect
    # sys.stdout to stderr so that any print() calls from sam3 (e.g.
    # emit_status, tqdm, logging) go to stderr instead of corrupting
    # the length-prefixed IPC protocol on the pipe.
    ipc_out = sys.stdout.buffer
    sys.stdout = sys.stderr

    worker = Sam3Worker()
    while True:
        request = recv_msg(sys.stdin.buffer)
        if request is None:
            break
        try:
            response = worker.handle(request)
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        send_msg(ipc_out, response)
        if response.get("exit"):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
