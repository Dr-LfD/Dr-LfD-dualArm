import os
import sys

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
    # Allow `from examples.pybullet...` when PYTHONPATH is unset (e.g. IDE Run/Debug).
    root = os.path.normpath(resolve_workspace_root(__file__).rstrip(os.sep))
    if not any(os.path.normpath(p.rstrip(os.sep)) == root for p in sys.path):
        sys.path.insert(0, root)


_ensure_pddlstream_aloha_on_syspath()


# Keys forwarded from effect_detection_defaults into a skill's effect_detection block.
CONTACT_PREDICTOR_DEFAULT_KEYS = ("text_prompt", "contact_label_threshold")


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

        from contact_pred.scripts.common.contact_predictor_runtime import (
            ContactPredictorRuntimeSession,
        )

        self.session = ContactPredictorRuntimeSession(
            device=self.device,
            output_dir=self.output_dir,
        )
        self._skill_state = {}
        self._active_skill_keys = []

    def predict_effect(self, skill_meta):
        effect_cfg = self._validate_effect_detection(skill_meta)
        skill_key = skill_meta.get("skill_name")
        state = self._skill_state.setdefault(
            skill_key,
            {"frame_index": 0, "last_prediction": None, "effect_cfg": effect_cfg},
        )
        state["effect_cfg"] = effect_cfg
        if state["last_prediction"] is not None and state["frame_index"] > 0:
            return state["last_prediction"]

        prediction = self._predict_multi(effect_cfg)
        state["last_prediction"] = prediction
        return prediction

    @staticmethod
    def _effect_edges_from_effect_cfg(effect_cfg):
        edge_text = effect_cfg.get("edge_text")
        if isinstance(edge_text, str) and edge_text.strip():
            return [x.strip() for x in edge_text.split(",") if x.strip()]
        if isinstance(edge_text, (list, tuple)) and edge_text:
            return [str(x) for x in edge_text]
        if effect_cfg.get("contact_edge"):
            return [effect_cfg["contact_edge"]]
        raise ValueError("effect_detection must set contact_edge or a non-empty edge_text.")

    def _predict_multi(self, effect_cfg):
        rule = effect_cfg.get("multi_edge_rule", "any")
        predictions = []
        for edge in self._effect_edges_from_effect_cfg(effect_cfg):
            edge_effect_cfg = {**effect_cfg, "contact_edge": edge}
            runtime = self._get_runtime(edge_effect_cfg)
            predictions.append(self._predict_label(runtime, edge_effect_cfg))
        return all(predictions) if rule == "all" else any(predictions)

    def begin_monitoring(self, skill_metas):
        self._active_skill_keys = []
        for skill_meta in skill_metas:
            effect_cfg = self._validate_effect_detection(skill_meta)
            skill_key = skill_meta.get("skill_name")
            self._skill_state[skill_key] = {
                "frame_index": 0,
                "last_prediction": None,
                "effect_cfg": effect_cfg,
                "skill_meta": skill_meta,
            }
            self._active_skill_keys.append(skill_key)

    def end_monitoring(self):
        self._active_skill_keys = []

    def observe_frame(self):
        for skill_key in list(self._active_skill_keys):
            state = self._skill_state[skill_key]
            effect_cfg = state["effect_cfg"]
            if state["frame_index"] % effect_cfg["skip_frame"] == 0:
                state["last_prediction"] = self._predict_multi(effect_cfg)
            state["frame_index"] += 1

    def _validate_effect_detection(self, skill_meta):
        effect_cfg = dict(skill_meta.get("effect_detection"))
        defaults = self._effect_detection_defaults
        if "checkpoint" not in effect_cfg and defaults.get("contact_predictor_checkpoint"):
            effect_cfg["checkpoint"] = defaults["contact_predictor_checkpoint"]
        for key in CONTACT_PREDICTOR_DEFAULT_KEYS:
            if key not in effect_cfg and defaults.get(key) is not None:
                effect_cfg[key] = defaults[key]
        if effect_cfg.get("backend") != "contact_predictor":
            raise ValueError(
                f"skill '{skill_meta.get('skill_name', skill_meta)}' is not configured for the contact predictor backend."
            )
        if not effect_cfg.get("checkpoint"):
            raise ValueError("effect_detection['checkpoint'] is required for contact_predictor.")
        if not effect_cfg.get("contact_edge") and not effect_cfg.get("edge_text"):
            raise ValueError(
                "effect_detection['contact_edge'] or effect_detection['edge_text'] is required "
                "for contact_predictor."
            )
        if not effect_cfg.get("contact_edge"):
            effect_cfg["contact_edge"] = self._effect_edges_from_effect_cfg(effect_cfg)[0]
        skip_frame = effect_cfg.get("skip_frame", 1)
        if not isinstance(skip_frame, int) or skip_frame < 1:
            raise ValueError("effect_detection['skip_frame'] must be an integer >= 1.")
        effect_cfg["skip_frame"] = skip_frame
        effect_cfg["checkpoint"] = self._resolve_checkpoint_path(effect_cfg["checkpoint"])
        return effect_cfg

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

    def _get_runtime(self, effect_cfg):
        checkpoint = effect_cfg["checkpoint"]
        runtime = self.session.load_runtime(checkpoint)
        expected_edge = runtime.get("contact_edge")
        if expected_edge and expected_edge != effect_cfg["contact_edge"]:
            raise ValueError(
                f"Checkpoint '{checkpoint}' is configured for contact_edge '{expected_edge}', "
                f"but effect_detection requested '{effect_cfg['contact_edge']}'."
            )
        return runtime

    def _predict_label(self, runtime, effect_cfg):
        from contact_pred.scripts.common.contact_predictor_runtime import LFDObservationProvider

        obs_dict = self.session.build_obs_dict(
            runtime,
            effect_cfg,
            LFDObservationProvider(self.lfd),
            default_obs_key_map=self.DEFAULT_OBS_KEY_MAP,
        )
        score = float(self.session.predict_label(runtime, obs_dict))
        threshold = effect_cfg.get("contact_label_threshold")
        if threshold is None:
            threshold = runtime.get("contact_label_threshold", 0.5)
        return score > float(threshold)


class ContactEffectMonitor:
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
    effect_cfg = skill_meta.get("effect_detection")
    if effect_cfg.get("backend") == "contact_predictor":
        if contact_predictor_wrapper is None:
            raise ValueError(
                f"contact predictor wrapper is required for skill '{skill_meta.get('skill_name')}'."
            )
        return ContactEffectMonitor(contact_predictor_wrapper, [skill_meta])

    from examples.pybullet.aloha_real.openworld_aloha.symbolic_utils import BiopCompletionMonitor

    return BiopCompletionMonitor(skill_meta, env_type)


class _ConstantImageObservation(dict):
    # Return the same camera image for any requested observation key.
    def __init__(self, image):
        super().__init__()
        self._image = image

    def get(self, key, default=None):
        return self._image

    def __getitem__(self, key):
        return self._image


def _lfd_from_image(image_hwc_uint8):
    # Minimal `lfd` using ts.observation (CHW float) to match deployment/rollout without raw flip.
    from types import SimpleNamespace

    img = np.asarray(image_hwc_uint8, dtype=np.float32) / 255.0
    chw = np.moveaxis(img, -1, 0)
    return SimpleNamespace(ts=SimpleNamespace(observation=_ConstantImageObservation(chw)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test the contact predictor on a single wrist image."
    )
    parser.add_argument("--image", required=True, help="Wrist-camera RGB image to score.")
    parser.add_argument("--checkpoint", required=True, help="Contact predictor .ckpt.")
    parser.add_argument("--contact-edge", default="left_arm_screwdriver")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument(
        "--contact-prediction-root", default=None,
        help="ContactPrediction checkout (default: $CONTACT_PREDICTION_ROOT, "
             "else the vendored <repo>/contact_detector submodule).",
    )
    args = parser.parse_args()

    # CLI paths bypass load_yaml's ${VAR}/~ expansion.
    def _expand(p):
        return os.path.expanduser(os.path.expandvars(p)) if p else p

    IMAGE = _expand(args.image)
    CONTACT_EDGE = args.contact_edge
    DEVICE = args.device
    THRESHOLD = args.threshold

    CONTACT_REPO = (
        _expand(args.contact_prediction_root)
        or os.environ.get("CONTACT_PREDICTION_ROOT")
        or os.path.join(resolve_workspace_root(__file__), "contact_detector")
    )
    if not os.path.isdir(os.path.join(CONTACT_REPO, "contact_pred")):
        raise SystemExit(
            f"ContactPrediction package not found under {CONTACT_REPO!r}; set "
            "--contact-prediction-root or $CONTACT_PREDICTION_ROOT, or initialize "
            "the contact_detector submodule."
        )
    CHECKPOINT = _expand(args.checkpoint)
    OUTPUT_DIR = os.path.join(os.getcwd(), "_contact_monitor_debug")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    effect_detection_defaults = {"contact_label_threshold": THRESHOLD}

    sys.path.insert(0, CONTACT_REPO)
    from contact_pred.scripts.common.contact_predictor_runtime import (
        ContactPredictorRuntimeSession,
        LFDObservationProvider,
    )

    checkpoint_abs = os.path.abspath(CHECKPOINT)

    session = ContactPredictorRuntimeSession(device=DEVICE, output_dir=OUTPUT_DIR)
    runtime = session.load_runtime(checkpoint_abs)

    import cv2

    bgr = cv2.imread(os.path.expanduser(IMAGE))
    image_hwc = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    lfd = _lfd_from_image(image_hwc)

    skill_meta = {
        "skill_name": "__debug_image__",
        "effect_detection": {
            "backend": "contact_predictor",
            "checkpoint": checkpoint_abs,
            "contact_edge": CONTACT_EDGE,
            "skip_frame": 1,
            "contact_label_threshold": THRESHOLD,
        },
    }

    wrapper = ContactPredictorWrapper(
        lfd,
        output_dir=OUTPUT_DIR,
        contact_repo_root=CONTACT_REPO,
        device=DEVICE,
        workspace_root=os.getcwd(),
        effect_detection_defaults=effect_detection_defaults,
    )
    monitor = ContactEffectMonitor(wrapper, [skill_meta])
    effect_cfg = wrapper._validate_effect_detection(skill_meta)
    runtime_live = wrapper._get_runtime(effect_cfg)
    obs_dict = session.build_obs_dict(
        runtime_live, effect_cfg, LFDObservationProvider(lfd),
        default_obs_key_map=ContactPredictorWrapper.DEFAULT_OBS_KEY_MAP,
    )
    score = float(session.predict_label(runtime_live, obs_dict))
    print(f"label={score}")
    print(f"binary(label>{THRESHOLD})={score > THRESHOLD}")
    print(f"monitor.update()={monitor.update()}")
    monitor.close()
