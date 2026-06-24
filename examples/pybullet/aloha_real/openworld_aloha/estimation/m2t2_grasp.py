"""In-process M2T2 grasp predictor for scene point clouds.

The generic (geometry-only) M2T2 model is loaded once and predicts 6-DoF grasp
poses from a scene point cloud expressed in the pybullet/world frame. M2T2 lives
in a sibling repository; its package root is prepended to ``sys.path`` here so
the TAMP conda env (``mj_lfd_tamp``) can import it directly -- it already ships a
matching torch build and the compiled ``pointnet2_ops``, so no worker process is
needed.

Mirrors the data construction of ``M2T2/predict_ply.py`` but drops the Hydra
entry point in favour of an explicit ``OmegaConf.load`` so it can be driven from
an arbitrary caller.
"""

import sys

import numpy as np
import torch
from omegaconf import OmegaConf

# ImageNet statistics used by the M2T2 training pipeline. RGB is only carried to
# keep the input width at 6; the scene encoder runs geometry-only (use_rgb=False),
# so colorless scenes are fed zeros without affecting predictions.
_RGB_MEAN = torch.tensor([0.485, 0.456, 0.406])
_RGB_STD = torch.tensor([0.229, 0.224, 0.225])


class M2T2GraspPredictor:
    """Thin wrapper around an M2T2 model that returns ranked world-frame grasps."""

    def __init__(
        self,
        checkpoint,
        config_path,
        repo_root,
        mask_thresh=None,
        num_points=None,
        num_runs=1,
        visualize=False,
        viz_top_k=20,
        viz_pause=True,
    ):
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from m2t2.m2t2 import M2T2
        from m2t2.dataset import collate
        from m2t2.dataset_utils import sample_points
        from m2t2.train_utils import to_cpu, to_gpu

        self._collate = collate
        self._sample_points = sample_points
        self._to_cpu = to_cpu
        self._to_gpu = to_gpu

        cfg = OmegaConf.load(config_path)
        if mask_thresh is not None:
            cfg.eval.mask_thresh = float(mask_thresh)
        if num_points is not None:
            cfg.data.num_points = int(num_points)
        cfg.eval.num_runs = int(num_runs)
        cfg.eval.checkpoint = checkpoint
        self.cfg = cfg

        model = M2T2.from_config(cfg.m2t2)
        ckpt = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        # Single-GPU pipeline (CUDA_VISIBLE_DEVICES selects the device); match
        # M2T2's to_gpu/to_cpu helpers which hardcode .cuda()/.cpu().
        self.model = model.cuda().eval()

        # Optional meshcat preview of predicted grasps. The visualizer is created
        # lazily on first use so non-visual inference never imports meshcat or
        # connects to a server.
        self._visualize = bool(visualize)
        self._viz_top_k = int(viz_top_k)
        self._viz_pause = bool(viz_pause)
        self._vis = None

    def _build_data(self, xyz_np, rgb_np):
        """Replicate ``predict_ply.build_data_from_ply`` for a pick scene."""
        xyz = torch.from_numpy(np.asarray(xyz_np, dtype=np.float32))
        if rgb_np is not None:
            rgb = (torch.from_numpy(np.asarray(rgb_np, dtype=np.float32)) - _RGB_MEAN) / _RGB_STD
        else:
            rgb = torch.zeros(len(xyz), 3)
        return {
            "inputs": torch.cat([xyz - xyz.mean(dim=0), rgb], dim=1),
            "points": xyz,
            "seg": torch.zeros(len(xyz), dtype=torch.long),
            "cam_pose": torch.eye(4),
            "object_inputs": torch.zeros(1024, 6),
            "ee_pose": torch.eye(4),
            "bottom_center": torch.zeros(3),
            "object_center": torch.zeros(3),
            "task": "pick",
        }

    @torch.no_grad()
    def predict(self, scene_xyz, scene_rgb=None):
        """Predict grasps for a scene point cloud.

        Args:
            scene_xyz: (N, 3) array of points in the pybullet/world frame.
            scene_rgb: optional (N, 3) array of colors in [0, 1].

        Returns:
            List of dicts ``{'pose': (4, 4), 'confidence': float, 'contact': (3,)}``
            in the input frame (``world_coord=True``), ranked by descending
            confidence.
        """
        data = self._build_data(scene_xyz, scene_rgb)
        inputs_full = data["inputs"]
        xyz_full = data["points"]
        seg_full = data["seg"]
        obj_full = data["object_inputs"]

        ranked = []
        for _ in range(int(self.cfg.eval.num_runs)):
            pt_idx = self._sample_points(xyz_full, self.cfg.data.num_points)
            data["inputs"] = inputs_full[pt_idx]
            data["points"] = xyz_full[pt_idx]
            data["seg"] = seg_full[pt_idx]
            data["object_inputs"] = obj_full

            batch = self._collate([data])
            self._to_gpu(batch)
            model_out = self.model.infer(batch, self.cfg.eval)
            self._to_cpu(model_out)

            # infer returns per-batch lists; index 0 is our single scene, whose
            # entries are one (Ng, 4, 4)/(Ng,)/(Ng, 3) tensor per detected object.
            for obj_grasps, obj_conf, obj_contacts in zip(
                model_out["grasps"][0],
                model_out["grasp_confidence"][0],
                model_out["grasp_contacts"][0],
            ):
                for pose, conf, contact in zip(
                    obj_grasps.numpy(), obj_conf.numpy(), obj_contacts.numpy()
                ):
                    ranked.append(
                        {
                            "pose": pose.astype(np.float64),
                            "confidence": float(conf),
                            "contact": contact.astype(np.float64),
                        }
                    )

        ranked.sort(key=lambda d: d["confidence"], reverse=True)

        if self._visualize and ranked:
            self._preview(scene_xyz, scene_rgb, ranked)

        return ranked

    def _preview(self, scene_xyz, scene_rgb, ranked):
        """Draw the scene cloud and top-K predicted grasps on M2T2's meshcat.

        Grasps and ``scene_xyz`` share the world frame (``world_coord=True``), so
        the gripper wireframes overlay the cloud without any transform. Imports
        are deferred here so ``meshcat`` is only required when visualization is on.
        Requires a meshcat server on ``tcp://127.0.0.1:6000`` (run ``meshcat-server``).
        """
        from m2t2.meshcat_utils import (
            create_visualizer,
            visualize_grasp,
            visualize_pointcloud,
        )

        if self._vis is None:
            self._vis = create_visualizer()
        else:
            self._vis.delete()

        xyz = np.asarray(scene_xyz, dtype=np.float32)
        if scene_rgb is not None:
            # visualize_pointcloud expects [0, 255]; our inputs are in [0, 1].
            rgb_u8 = (np.asarray(scene_rgb, dtype=np.float32) * 255).astype(np.uint8)
        else:
            rgb_u8 = np.full((len(xyz), 3), 200, dtype=np.uint8)
        visualize_pointcloud(self._vis, "scene", xyz, rgb_u8, size=0.005)

        for i, cand in enumerate(ranked[: self._viz_top_k]):
            conf = cand["confidence"]
            # Green for high confidence, red for low; integer RGB in [0, 255].
            color = [int((1.0 - conf) * 255), int(conf * 255), 0]
            visualize_grasp(self._vis, f"grasp/{i:03d}", cand["pose"], color, linewidth=2)

        if self._viz_pause:
            input("[m2t2] grasps drawn -- press Enter to continue...")
