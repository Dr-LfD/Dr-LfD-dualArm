import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np


def resolve_workspace_root(current_file):
    current = os.path.dirname(os.path.abspath(current_file))
    while not os.path.isfile(os.path.join(current, ".repo_root")):
        parent = os.path.dirname(current)
        if parent == current:
            raise ValueError(
                f"Unable to resolve workspace root from '{current_file}': "
                "no .repo_root marker found in any parent directory."
            )
        current = parent
    return current


def _ensure_pddlstream_aloha_on_syspath():
    """Allow ``from examples.pybullet...`` when PYTHONPATH is unset (e.g. IDE Run/Debug)."""
    root = os.path.normpath(resolve_workspace_root(__file__).rstrip(os.sep))
    if not any(os.path.normpath(p.rstrip(os.sep)) == root for p in sys.path):
        sys.path.insert(0, root)


_ensure_pddlstream_aloha_on_syspath()


_VALID_SEG_BACKENDS = frozenset({"sam3", "grounded_sam"})


def canonicalize_seg_backend(raw, *, field_name: str, mirror: Optional[str] = None) -> str:
    """Map config to sam3 | grounded_sam. mirror used when raw is null/empty."""
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        if mirror is None:
            raise ValueError(f"{field_name} is required.")
        return mirror
    value = str(raw).strip()
    if value == "sam":
        raise ValueError(
            f"{field_name}: 'sam' is no longer supported; use 'grounded_sam'."
        )
    if value in _VALID_SEG_BACKENDS:
        return value
    raise ValueError(
        f"{field_name} must be 'sam3' or 'grounded_sam'; got {value!r}."
    )


def init_seg_branch_for_canonical(canonical: str) -> str:
    """init_seg / estimation_policy branch id (legacy sam class for GSAM)."""
    if canonical == "grounded_sam":
        return "sam"
    if canonical == "sam3":
        return "sam3"
    raise ValueError(
        f"canonical seg backend must be 'sam3' or 'grounded_sam'; got {canonical!r}."
    )


@dataclass(frozen=True)
class SegBackendPairing:
    perception: str
    contact: str
    reuse_mode: str


def validate_seg_backend_pairing(para) -> SegBackendPairing:
    perception = canonicalize_seg_backend(
        para.get("seg_branch", "sam3"), field_name="seg_branch"
    )
    contact = canonicalize_seg_backend(
        para.get("contact_wrist_seg_backend"),
        field_name="contact_wrist_seg_backend",
        mirror=perception,
    )
    if perception == contact:
        return SegBackendPairing(perception, contact, reuse_mode=perception)
    raise ValueError(
        f"Perception seg backend {perception!r} and contact wrist seg backend "
        f"{contact!r} must match (set contact_wrist_seg_backend: null to mirror perception)."
    )


def build_seg_registry_para(para, pairing: SegBackendPairing) -> dict:
    """Fill missing sam3 / grounded_sam paths from ContactPrediction segmentation.yaml."""
    merged = dict(para or {})
    contact_repo = merged.get("contact_prediction_root")
    if not contact_repo:
        return merged
    contact_repo = os.path.abspath(os.path.expanduser(contact_repo))
    merged.setdefault("contact_prediction_root", contact_repo)
    if contact_repo not in sys.path:
        sys.path.insert(0, contact_repo)
    from contact_pred.scripts.common.segmentation_config import load_segmentation_config

    cfg = load_segmentation_config()
    if pairing.reuse_mode == "sam3":
        sam3 = dict(cfg.get("sam3") or {})
        merged.setdefault("sam3_worker_path", sam3.get("worker_script"))
        merged.setdefault("sam3_path", sam3.get("sam3_path"))
        merged.setdefault("sam3_model_dir", sam3.get("model_dir"))
        merged.setdefault("sam3_checkpoint", sam3.get("checkpoint"))
        merged.setdefault("sam3_conda_env", sam3.get("conda_env"))
        merged.setdefault("sam3_conda_bin", sam3.get("conda_bin"))
    elif pairing.reuse_mode == "grounded_sam":
        gsam = dict(cfg.get("grounded_sam") or {})
        merged.setdefault("sam_path", gsam.get("repo_root"))
    return merged


class SegReuseRegistry:
    _pairing: Optional[SegBackendPairing] = None
    _sam3_client = None
    _grounded_sam_segmenter = None

    @classmethod
    def init_for_pairing(cls, pairing: SegBackendPairing, para) -> None:
        para = build_seg_registry_para(para, pairing)
        cls._pairing = pairing
        cls._sam3_client = None
        cls._grounded_sam_segmenter = None
        if pairing.reuse_mode == "sam3":
            cls._sam3_client = cls._create_sam3_client(para)
        elif pairing.reuse_mode == "grounded_sam":
            cls._grounded_sam_segmenter = cls._create_grounded_sam_segmenter(para)

    @classmethod
    def pairing(cls) -> SegBackendPairing:
        if cls._pairing is None:
            raise RuntimeError("SegReuseRegistry.init_for_pairing was not called.")
        return cls._pairing

    @classmethod
    def get_sam3_client(cls):
        if cls.pairing().reuse_mode != "sam3":
            raise RuntimeError(
                f"SAM3 client is not active (reuse_mode={cls.pairing().reuse_mode!r})."
            )
        if cls._sam3_client is None:
            raise RuntimeError("SAM3 client was not initialized.")
        return cls._sam3_client

    @classmethod
    def get_grounded_sam_segmenter(cls):
        if cls.pairing().reuse_mode != "grounded_sam":
            raise RuntimeError(
                "GroundedSamSegmenter is not active "
                f"(reuse_mode={cls.pairing().reuse_mode!r})."
            )
        if cls._grounded_sam_segmenter is None:
            raise RuntimeError("GroundedSamSegmenter was not initialized.")
        return cls._grounded_sam_segmenter

    @staticmethod
    def _create_sam3_client(para):
        worker_path = para.get("sam3_worker_path")
        sam3_path = para.get("sam3_path")
        if not worker_path or not sam3_path:
            raise ValueError(
                "seg_branch sam3 requires sam3_worker_path and sam3_path in config."
            )
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
        contact_repo = para.get("contact_prediction_root")
        if not contact_repo:
            raise ValueError(
                "seg_branch grounded_sam requires contact_prediction_root in config."
            )
        contact_repo = os.path.abspath(os.path.expanduser(contact_repo))
        if contact_repo not in sys.path:
            sys.path.insert(0, contact_repo)
        from contact_pred.scripts.common.grounded_sam_segmenter import GroundedSamSegmenter
        from contact_pred.scripts.common.segmentation_config import (
            get_grounded_sam_config,
            load_segmentation_config,
        )

        gsam_cfg = dict(load_segmentation_config()["grounded_sam"])
        sam_path = para.get("sam_path")
        if sam_path:
            gsam_cfg["repo_root"] = os.path.abspath(os.path.expanduser(sam_path))
        resolved = get_grounded_sam_config(gsam_cfg)
        repo_root = resolved["repo_root"]
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        return GroundedSamSegmenter(resolved)


class _Sam3MaskClientAdapter:
    def __init__(self, client):
        self._client = client

    def segment(self, rgb_image, class_names):
        return self._client.segment(rgb_image, class_names)


class _GroundedSamMaskClientAdapter:
    def __init__(self, segmenter):
        self._segmenter = segmenter

    def segment(self, rgb_image, class_names):
        rgb = np.asarray(rgb_image, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected HxWx3 RGB, got shape {rgb.shape}.")
        masks_by_class = {}
        for name in class_names:
            mask_seq = self._segmenter.segment_sequence(rgb[np.newaxis], name)
            mask = np.asarray(mask_seq[0], dtype=bool)
            masks_by_class[name] = [mask] if mask.any() else []
        return masks_by_class


def _resolve_edge_object_name(edge_name, robot_camera_map):
    # ## if cup is near, only 'plastic' can do
    # return 'plastic'
    """Infer SAM3 class name (e.g. ``cup``) from ``{arm}_{obj}`` using longest robot prefix match."""
    matched_robot = next(
        (
            robot_name
            for robot_name in sorted(robot_camera_map, key=len, reverse=True)
            if edge_name == robot_name or edge_name.startswith(f"{robot_name}_")
        ),
        None,
    )
    if matched_robot is None:
        raise ValueError(
            f"Cannot infer object name from edge '{edge_name}' using robot_camera_map "
            f"keys {sorted(robot_camera_map.keys())}."
        )
    object_name = edge_name[len(matched_robot) + 1 :]
    if not object_name:
        raise ValueError(
            f"Edge '{edge_name}' has no object suffix after robot '{matched_robot}'."
        )
    return object_name


class _MaskedLFDProvider:
    """Apply wrist foreground mask to RGB before contact predictor preprocessing."""

    def __init__(self, base_provider, mask_client, object_name, apply_mask_fn):
        self._base = base_provider
        self._mask_client = mask_client
        self._object_name = object_name
        self._apply_mask = apply_mask_fn

    def get_image(self, obs_key):
        image = np.asarray(self._base.get_image(obs_key))
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"Masked wrist requires HxWx3 RGB, got shape {image.shape} for '{obs_key}'."
            )
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and float(np.max(image)) <= 1.0:
                image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        masks_by_class = self._mask_client.segment(image, [self._object_name])
        return self._apply_mask(image, masks_by_class, self._object_name)


class ContactPredictorWrapper:
    DEFAULT_OBS_KEY_MAP = {
        "cam_robot0": "robot0_eye_in_hand_image",
        "cam_robot1": "robot1_eye_in_hand_image",
        "cam_agent": "agentview_image",
    }

    def __init__(
        self,
        lfd,
        output_dir=None,
        contact_repo_root=None,
        device=None,
        workspace_root=None,
        effect_detection_defaults=None,
        seg_pairing=None,
    ):
        self.lfd = lfd
        self.output_dir = output_dir or os.getcwd()
        if not contact_repo_root:
            raise ValueError(
                "contact_prediction_root is required for ContactPredictorWrapper. "
                "Set it in sgBase.yaml and pass it as contact_repo_root."
            )
        self.contact_repo_root = os.path.abspath(os.path.expanduser(contact_repo_root))
        if not os.path.isdir(self.contact_repo_root):
            raise FileNotFoundError(
                f"contact_prediction_root does not exist: {self.contact_repo_root}"
            )
        if self.contact_repo_root not in sys.path:
            sys.path.insert(0, self.contact_repo_root)
        self.device = device
        self.workspace_root = workspace_root or os.getcwd()
        self._effect_detection_defaults = dict(effect_detection_defaults or {})
        self._seg_pairing = seg_pairing or SegReuseRegistry.pairing()
        self._runtime_session = None
        self._skill_state = {}
        self._active_skill_keys = []

    def _get_wrist_mask_client(self):
        reuse_mode = self._seg_pairing.reuse_mode
        if reuse_mode == "sam3":
            return _Sam3MaskClientAdapter(SegReuseRegistry.get_sam3_client())
        if reuse_mode == "grounded_sam":
            return _GroundedSamMaskClientAdapter(
                SegReuseRegistry.get_grounded_sam_segmenter()
            )
        raise RuntimeError(f"Unknown wrist mask reuse_mode {reuse_mode!r}.")

    def _get_image_provider(self, runtime, det):
        from contact_pred.scripts.common.contact_predictor_runtime import LFDObservationProvider

        base = LFDObservationProvider(self.lfd)

        # if runtime.get("use_wrist_bbox"):
        #     raise ValueError(
        #         "use_wrist_bbox checkpoints are not supported in open-world contact monitoring. "
        #         "Use a plain-wrist or use_masked_wrist checkpoint, or run rollout eval with bbox caching."
        #     )

        if not bool(runtime.get("use_masked_wrist", False)):
            return base, False

        from examples.pybullet.aloha_real.openworld_aloha.workers.sam3_mask_client import (
            apply_object_mask,
        )

        robot_camera_map = dict(runtime.get("robot_camera_map") or {})
        det_map = det.get("robot_camera_map")
        if isinstance(det_map, dict):
            robot_camera_map.update(det_map)
        if not robot_camera_map:
            raise ValueError(
                "use_masked_wrist checkpoints require a non-empty robot_camera_map "
                "(from checkpoint runtime and/or effect_detection)."
            )
        edge = det.get("contact_edge")
        if not edge:
            raise ValueError(
                "contact_edge is required for masked-wrist object resolution."
            )
        object_name = _resolve_edge_object_name(edge, robot_camera_map)
        # object_name = 'screwler'
        mask_client = self._get_wrist_mask_client()
        return _MaskedLFDProvider(base, mask_client, object_name, apply_object_mask), True

    def _get_runtime_session(self):
        if self._runtime_session is None:
            from contact_pred.scripts.common.contact_predictor_runtime import (
                ContactPredictorRuntimeSession,
            )

            self._runtime_session = ContactPredictorRuntimeSession(
                device=self.device,
                output_dir=self.output_dir,
            )
        return self._runtime_session

    def predict_effect(self, skill_meta):
        det = self._validate_effect_detection(skill_meta)
        skill_key = skill_meta.get("skill_name")
        state = self._skill_state.setdefault(
            skill_key,
            {"frame_index": 0, "last_prediction": None, "det": det},
        )
        state["det"] = det
        if state["last_prediction"] is not None and state["frame_index"] > 0:
            return state["last_prediction"]

        prediction = self._predict_multi(det)
        state["last_prediction"] = prediction
        return prediction

    @staticmethod
    def _effect_edges_from_det(det):
        et = det.get("edge_text")
        if isinstance(et, str) and et.strip():
            return [x.strip() for x in et.split(",") if x.strip()]
        if isinstance(et, (list, tuple)) and et:
            return [str(x) for x in et]
        if det.get("contact_edge"):
            return [det["contact_edge"]]
        raise ValueError("effect_detection must set contact_edge or a non-empty edge_text.")

    def _predict_multi(self, det):
        rule = det.get("multi_edge_rule", "any")
        preds = []
        for edge in self._effect_edges_from_det(det):
            sub = {**det, "contact_edge": edge}
            runtime = self._get_runtime(sub)
            preds.append(self._predict_label(runtime, sub))
        return all(preds) if rule == "all" else any(preds)

    def begin_monitoring(self, skill_metas):
        self._active_skill_keys = []
        for skill_meta in skill_metas:
            det = self._validate_effect_detection(skill_meta)
            skill_key = skill_meta.get("skill_name")
            self._skill_state[skill_key] = {
                "frame_index": 0,
                "last_prediction": None,
                "det": det,
                "skill_meta": skill_meta,
            }
            self._active_skill_keys.append(skill_key)

    def end_monitoring(self):
        self._active_skill_keys = []

    def observe_frame(self):
        for skill_key in list(self._active_skill_keys):
            state = self._skill_state[skill_key]
            det = state["det"]
            if state["frame_index"] % det["skip_frame"] == 0:
                state["last_prediction"] = self._predict_multi(det)
            state["frame_index"] += 1

    def _validate_effect_detection(self, skill_meta):
        det = dict(skill_meta.get("effect_detection"))
        defaults = self._effect_detection_defaults
        if "checkpoint" not in det and defaults.get("contact_predictor_checkpoint"):
            det["checkpoint"] = defaults["contact_predictor_checkpoint"]
        for key in (
            "sam3_worker_path",
            "sam3_path",
            "sam3_model_dir",
            "sam3_checkpoint",
            "sam3_conda_env",
            "sam3_conda_bin",
            "sam_path",
            # "contact_label_threshold",
            "seg_branch",
            "contact_wrist_seg_backend",
            "text_prompt",
        ):
            if key not in det and key in defaults:
                det[key] = defaults[key]
        if det.get("backend") != "contact_predictor":
            raise ValueError(
                f"skill '{skill_meta.get('skill_name', skill_meta)}' is not configured for the contact predictor backend."
            )
        if not det.get("checkpoint"):
            raise ValueError("effect_detection['checkpoint'] is required for contact_predictor.")
        if not det.get("contact_edge") and not det.get("edge_text"):
            raise ValueError(
                "effect_detection['contact_edge'] or effect_detection['edge_text'] is required "
                "for contact_predictor."
            )
        if not det.get("contact_edge"):
            det["contact_edge"] = self._effect_edges_from_det(det)[0]
        skip_frame = det.get("skip_frame", 1)
        if not isinstance(skip_frame, int) or skip_frame < 1:
            raise ValueError("effect_detection['skip_frame'] must be an integer >= 1.")
        det["skip_frame"] = skip_frame
        det["checkpoint"] = self._resolve_checkpoint_path(det["checkpoint"])
        return det

    def _resolve_checkpoint_path(self, checkpoint):
        if os.path.isabs(checkpoint):
            if not os.path.exists(checkpoint):
                raise FileNotFoundError(f"Contact predictor checkpoint not found: {checkpoint}")
            return checkpoint

        candidates = [
            os.path.join(self.workspace_root, checkpoint),
            os.path.join(self.contact_repo_root, checkpoint),
            os.path.abspath(checkpoint),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.normpath(candidate)
        raise FileNotFoundError(f"Contact predictor checkpoint not found: {checkpoint}")

    def _get_runtime(self, det):
        checkpoint = det["checkpoint"]
        runtime = self._get_runtime_session().load_runtime(checkpoint)
        expected_edge = runtime.get("contact_edge")
        if expected_edge and expected_edge != det["contact_edge"]:
            raise ValueError(
                f"Checkpoint '{checkpoint}' is configured for contact_edge '{expected_edge}', "
                f"but effect_detection requested '{det['contact_edge']}'."
            )
        return runtime

    def _build_obs_dict(self, runtime, det):
        obs_dict = {}
        obs_shape_meta = runtime.get("shape_meta").get("obs")
        for obs_name, attr in obs_shape_meta.items():
            obs_type = attr.get("type")
            if obs_type == "rgb":
                obs_dict[obs_name] = self._build_image_obs(obs_name, runtime, det)
            elif obs_name == "edge_text":
                obs_dict[obs_name] = self._build_edge_text_obs(runtime, det, attr)
            else:
                raise KeyError(
                    f"Contact predictor checkpoint requires unsupported low-dim observation "
                    f"'{obs_name}'. Add runtime construction for this key before inference."
                )
        return obs_dict

    def _build_image_obs(self, camera_name, runtime, det):
        provider, masked_attested = self._get_image_provider(runtime, det)
        return self._get_runtime_session().build_image_obs(
            runtime,
            det,
            camera_name,
            provider,
            default_obs_key_map=self.DEFAULT_OBS_KEY_MAP,
            masked_wrist_applied=masked_attested,
        )

    def _build_edge_text_obs(self, runtime, det, attr):
        # embedding = np.asarray(
            # self._get_edge_text_embedding(runtime, det),
            # dtype=np.float32,
        # )
        embedding = self._get_runtime_session().get_edge_text_embedding(
            runtime, det["contact_edge"]
        )
        expected_shape = tuple(attr.get("shape"))
        if embedding.shape != expected_shape:
            embedding = embedding.reshape(expected_shape)
        return np.expand_dims(embedding, axis=0)

    def _get_edge_text_embedding(self, runtime, det):
        return self._get_runtime_session().get_edge_text_embedding(
            runtime, det["contact_edge"]
        )

    @staticmethod
    def _tensor_to_debug_uint8(arr):
        arr = np.asarray(arr, dtype=np.float32)
        while arr.ndim > 4:
            arr = arr[0]
        if arr.ndim == 4:
            arr = arr[-1]
        if arr.ndim != 3:
            return None
        if arr.shape[0] == 3:
            arr = np.moveaxis(arr, 0, -1)
        elif arr.shape[-1] != 3:
            return None
        lo, hi = float(arr.min()), float(arr.max())
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return (arr * 255.0).astype(np.uint8)

    def _save_predict_debug_images(self, runtime, obs_dict):
        import cv2
        obs_shape_meta = (runtime.get("shape_meta") or {}).get("obs") or {}
        repo_root = resolve_workspace_root(__file__).rstrip(os.sep)
        for obs_name, attr in obs_shape_meta.items():
            if attr.get("type") != "rgb" or obs_name not in obs_dict:
                continue
            img_uint8 = self._tensor_to_debug_uint8(obs_dict[obs_name])
            if img_uint8 is None:
                continue
            debug_path = os.path.join(repo_root, f"debug_predict_{obs_name}.png")
            cv2.imwrite(debug_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
    
    def _predict_label(self, runtime, det):
        obs_dict = self._build_obs_dict(runtime, det)
        self._save_predict_debug_images(runtime, obs_dict)
        raw = float(self._get_runtime_session().predict_label(runtime, obs_dict))
        # thr = float(
        #     det["contact_label_threshold"]
        #     if "contact_label_threshold" in det
        #     else runtime.get("contact_label_threshold", 0.5)
        # )
        thr = 0.28
        return raw > thr


class ContactEffectMonitor:
    def __init__(self, predictor_wrapper, skill_meta):
        self.predictor_wrapper = predictor_wrapper
        self.skill_meta = skill_meta
        self.predictor_wrapper.begin_monitoring([skill_meta])

    def update(self, _sensor_data=None):
        self.predictor_wrapper.observe_frame()
        return self.predictor_wrapper.predict_effect(self.skill_meta)

    def close(self):
        self.predictor_wrapper.end_monitoring()


class ContactEffectMonitorGroup:
    def __init__(self, predictor_wrapper, skill_metas):
        self.predictor_wrapper = predictor_wrapper
        self.skill_metas = list(skill_metas)
        self.predictor_wrapper.begin_monitoring(self.skill_metas)

    def update(self, sensor_data=None):
        del sensor_data  # unused; accepts dict for call-site compatibility with BiopCompletionMonitor
        self.predictor_wrapper.observe_frame()
        return all(
            self.predictor_wrapper.predict_effect(skill_meta)
            for skill_meta in self.skill_metas
        )

    def close(self):
        self.predictor_wrapper.end_monitoring()


def build_effect_monitor(skill_meta, contact_predictor_wrapper, env_type):
    if skill_meta is None or not skill_meta.get("effect_detection"):
        return None
    det = skill_meta.get("effect_detection")
    if det.get("backend") == "contact_predictor":
        if contact_predictor_wrapper is None:
            raise ValueError(
                f"contact predictor wrapper is required for skill '{skill_meta.get('skill_name')}'."
            )
        return ContactEffectMonitor(contact_predictor_wrapper, skill_meta)

    from examples.pybullet.aloha_real.openworld_aloha.symbolic_utils import BiopCompletionMonitor

    return BiopCompletionMonitor(skill_meta, env_type)


def _lfd_rgb_obs_keys_for_runtime(
    runtime, contact_edge, default_obs_key_map, resolve_source_camera_name
):
    """Mirror ``ContactPredictorRuntimeSession.build_image_obs`` obs_key resolution for each RGB slot."""
    det_stub = {"contact_edge": contact_edge}
    obs_shape_meta = (runtime.get("shape_meta") or {}).get("obs") or {}
    camera_names = list(runtime.get("camera_names") or [])
    robot_camera_map = dict(runtime.get("robot_camera_map") or {})
    obs_key_map = dict(default_obs_key_map or {})
    obs_key_map.update(det_stub.get("obs_key_map") or {})
    keys = []
    for camera_name, attr in obs_shape_meta.items():
        if attr.get("type") != "rgb":
            continue
        lookup_camera_name = camera_name
        if runtime.get("single_camera_per_edge") and camera_name not in obs_key_map:
            selected_camera_key = runtime.get("selected_camera_key")
            if selected_camera_key is not None and camera_name != selected_camera_key:
                raise KeyError(
                    f"Checkpoint requested image key '{camera_name}', but single_camera_per_edge "
                    f"expects '{selected_camera_key}'."
                )
            lookup_camera_name = resolve_source_camera_name(
                contact_edge, camera_names, robot_camera_map
            )
        obs_key = obs_key_map.get(lookup_camera_name, lookup_camera_name)
        keys.append(obs_key)
    return list(dict.fromkeys(keys))


def _lfd_from_image_and_runtime(image_hwc_uint8, runtime, contact_edge, resolve_source_camera_name):
    """Minimal ``lfd`` using ``ts.observation`` (CHW float) to match deployment / rollout without raw flip."""
    from types import SimpleNamespace

    obs_keys = _lfd_rgb_obs_keys_for_runtime(
        runtime, contact_edge, ContactPredictorWrapper.DEFAULT_OBS_KEY_MAP, resolve_source_camera_name
    )
    if not obs_keys:
        raise ValueError("Checkpoint shape_meta has no RGB observations to fill from --image.")
    img = np.asarray(image_hwc_uint8, dtype=np.float32) / 255.0
    chw = np.moveaxis(img, -1, 0)
    observation = {k: chw for k in obs_keys}
    return SimpleNamespace(ts=SimpleNamespace(observation=observation))


if __name__ == "__main__":
    import cv2

    _repo = resolve_workspace_root(__file__)
    IMAGE = os.path.expanduser("~/interbotix_ws/src/pddlstream_aloha/cam_wrist_rgb_latest.png")
    CONTACT_REPO = os.path.expandvars("${WS_ROOT}/ContactPrediction")
    # CHECKPOINT = "${WS_ROOT}/ContactPrediction/data/outputs/2026.05.11/14.45.50_train_keypose_transformer_handoff_cup_multi_edge_wristview/checkpoints/latest.ckpt"
    # CONTACT_EDGE = "right_arm_cup"
    CHECKPOINT = os.path.expandvars("${WS_ROOT}/ContactPrediction/data/outputs/2026.05.19/11.39.08_train_keypose_transformer_screwdriver_multi_edge_wristview/checkpoints/latest.ckpt")
    CONTACT_EDGE = "left_arm_screwdriver"
    DEVICE = "cuda:0"
    # THRESHOLD = 0.3
    SEG_BACKEND = "grounded_sam"  # "sam3" to use SAM3 worker subprocess
    OUTPUT_DIR = os.path.join(os.getcwd(), "_contact_monitor_debug")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # sam3 / grounded_sam paths from ContactPrediction segmentation.yaml
    # (contact_pred/scripts/config/segmentation.yaml) via contact_prediction_root.
    effect_detection_defaults = {
        "seg_branch": SEG_BACKEND,
        "contact_wrist_seg_backend": None,
        # "contact_label_threshold": THRESHOLD,
    }

    seg_pairing = validate_seg_backend_pairing(
        {
            "seg_branch": effect_detection_defaults["seg_branch"],
            "contact_wrist_seg_backend": effect_detection_defaults.get(
                "contact_wrist_seg_backend"
            ),
        }
    )
    registry_para = {
        "contact_prediction_root": CONTACT_REPO,
        "seg_branch": seg_pairing.perception,
    }
    SegReuseRegistry.init_for_pairing(seg_pairing, registry_para)
    print(f"seg_backend={seg_pairing.reuse_mode} (perception={seg_pairing.perception})")

    sys.path.insert(0, CONTACT_REPO)
    from contact_pred.scripts.common.contact_predictor_runtime import (
        ContactPredictorRuntimeSession,
        resolve_source_camera_name,
    )

    checkpoint_abs = os.path.abspath(CHECKPOINT)

    session = ContactPredictorRuntimeSession(
        device=DEVICE,
        output_dir=OUTPUT_DIR,
    )
    runtime = session.load_runtime(checkpoint_abs)

    bgr = cv2.imread(os.path.expanduser(IMAGE))
    image_hwc = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    lfd = _lfd_from_image_and_runtime(
        image_hwc, runtime, CONTACT_EDGE, resolve_source_camera_name
    )

    skill_meta = {
        "skill_name": "__debug_image__",
        "effect_detection": {
            "backend": "contact_predictor",
            "checkpoint": checkpoint_abs,
            "contact_edge": CONTACT_EDGE,
            "skip_frame": 1,
            # "contact_label_threshold": THRESHOLD,
        },
    }

    wrapper = ContactPredictorWrapper(
        lfd,
        output_dir=OUTPUT_DIR,
        contact_repo_root=CONTACT_REPO,
        device=DEVICE,
        workspace_root=os.getcwd(),
        effect_detection_defaults=effect_detection_defaults,
        seg_pairing=seg_pairing,
    )
    wrapper._runtime_session = session
    monitor = ContactEffectMonitor(wrapper, skill_meta)
    det = wrapper._validate_effect_detection(skill_meta)
    runtime_live = wrapper._get_runtime(det)
    raw = float(
        wrapper._get_runtime_session().predict_label(
            runtime_live, wrapper._build_obs_dict(runtime_live, det)
        )
    )
    print(f"label={raw}")
    print(f"binary(label>{THRESHOLD})={raw > THRESHOLD}")
    print(f"monitor.update()={monitor.update()}")
    monitor.close()