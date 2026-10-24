"""Reference motion clips for SONIC, loaded from the GEAR-SONIC ``robot_filtered`` pickles.

A pickle holds ``{clip_name: {"dof": (T, 29), "root_rot": (T, 4) xyzw, "root_trans_offset": (T, 3),
"pose_aa": (T, 30, 3), "fps": 30}}`` with joints in MuJoCo / hardware order. The policy consumes
clips at 50 Hz in Isaac Lab joint order, so this module resamples the same way the training
motion library does (linear in joint angle, slerp on the root quaternion, forward-difference
velocities at the target rate).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from . import quat
from .sonic_params import CONTROL_HZ, MUJOCO_TO_ISAACLAB


@dataclass
class ReferenceMotion:
    name: str
    fps: float
    qpos_isaac: np.ndarray    # (T, 29) joint positions, Isaac Lab order, absolute
    qvel_isaac: np.ndarray    # (T, 29) joint velocities, Isaac Lab order
    root_quat: np.ndarray     # (T, 4) pelvis orientation, (w, x, y, z), world frame
    root_pos: np.ndarray      # (T, 3) pelvis position, world frame (for visualisation only)

    @property
    def num_frames(self) -> int:
        return self.qpos_isaac.shape[0]

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps

    def clamp(self, frame: int) -> int:
        return int(min(max(frame, 0), self.num_frames - 1))

    @property
    def qpos_hw(self) -> np.ndarray:
        """Joint positions back in hardware order (for comparing against the sim)."""
        inv = np.empty_like(MUJOCO_TO_ISAACLAB)
        inv[MUJOCO_TO_ISAACLAB] = np.arange(len(MUJOCO_TO_ISAACLAB))
        return self.qpos_isaac[:, inv]


def _resample_indices(num_src: int, src_fps: float, dst_fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror of torch_humanoid_batch.interploate_pose / _compute_frame_blend."""
    duration = (num_src - 1) / src_fps
    times = np.arange(0.0, duration, 1.0 / dst_fps, dtype=np.float32)
    phase = times / duration
    i0 = np.floor(phase * (num_src - 1)).astype(int)
    i1 = np.minimum(i0 + 1, num_src - 1)
    blend = phase * (num_src - 1) - i0
    return i0, i1, blend.astype(np.float64)


def load_reference_motions(pkl_path: str | Path, target_fps: float = CONTROL_HZ) -> dict[str, ReferenceMotion]:
    raw = joblib.load(str(pkl_path))
    out: dict[str, ReferenceMotion] = {}
    for name, clip in raw.items():
        src_fps = float(clip.get("fps", 30.0))
        dof_hw = np.asarray(clip["dof"], dtype=np.float64)                 # (T, 29) hardware order
        root_quat = quat.xyzw_to_wxyz(np.asarray(clip["root_rot"], dtype=np.float64))
        root_pos = np.asarray(clip["root_trans_offset"], dtype=np.float64)

        if abs(src_fps - target_fps) > 1e-6:
            i0, i1, b = _resample_indices(dof_hw.shape[0], src_fps, target_fps)
            dof_hw = (1 - b)[:, None] * dof_hw[i0] + b[:, None] * dof_hw[i1]
            root_pos = (1 - b)[:, None] * root_pos[i0] + b[:, None] * root_pos[i1]
            root_quat = np.stack([quat.slerp(root_quat[a], root_quat[c], t) for a, c, t in zip(i0, i1, b)])

        qpos_isaac = dof_hw[:, MUJOCO_TO_ISAACLAB]
        dt = 1.0 / target_fps
        qvel = (qpos_isaac[1:] - qpos_isaac[:-1]) / dt
        qvel_isaac = np.concatenate([qvel, qvel[-1:]], axis=0)  # motion lib repeats the last velocity

        out[str(name)] = ReferenceMotion(
            name=str(name), fps=target_fps, qpos_isaac=qpos_isaac, qvel_isaac=qvel_isaac,
            root_quat=root_quat, root_pos=root_pos,
        )
    return out
