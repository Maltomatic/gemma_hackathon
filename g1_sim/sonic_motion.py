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


def make_reference_motion(
    name: str,
    dof_hw: np.ndarray,
    root_quat_wxyz: np.ndarray,
    root_pos: np.ndarray,
    src_fps: float,
    target_fps: float = CONTROL_HZ,
) -> ReferenceMotion:
    """Build a 50 Hz Isaac-Lab-ordered clip from hardware-order joint angles at ``src_fps``."""
    dof_hw = np.asarray(dof_hw, dtype=np.float64)
    root_quat = np.asarray(root_quat_wxyz, dtype=np.float64)
    root_pos = np.asarray(root_pos, dtype=np.float64)
    assert dof_hw.shape[1] == 29, dof_hw.shape

    if abs(src_fps - target_fps) > 1e-6:
        i0, i1, b = _resample_indices(dof_hw.shape[0], src_fps, target_fps)
        dof_hw = (1 - b)[:, None] * dof_hw[i0] + b[:, None] * dof_hw[i1]
        root_pos = (1 - b)[:, None] * root_pos[i0] + b[:, None] * root_pos[i1]
        root_quat = np.stack([quat.slerp(root_quat[a], root_quat[c], t) for a, c, t in zip(i0, i1, b)])

    qpos_isaac = dof_hw[:, MUJOCO_TO_ISAACLAB]
    dt = 1.0 / target_fps
    qvel = (qpos_isaac[1:] - qpos_isaac[:-1]) / dt
    qvel_isaac = np.concatenate([qvel, qvel[-1:]], axis=0)  # motion lib repeats the last velocity

    return ReferenceMotion(
        name=name, fps=target_fps, qpos_isaac=qpos_isaac, qvel_isaac=qvel_isaac,
        root_quat=root_quat, root_pos=root_pos,
    )


def load_reference_motions(pkl_path: str | Path, target_fps: float = CONTROL_HZ) -> dict[str, ReferenceMotion]:
    """GEAR-SONIC ``robot_filtered`` pickle: one or more clips, 30 fps, hardware joint order."""
    raw = joblib.load(str(pkl_path))
    return {
        str(name): make_reference_motion(
            name=str(name),
            dof_hw=clip["dof"],
            root_quat_wxyz=quat.xyzw_to_wxyz(np.asarray(clip["root_rot"], dtype=np.float64)),
            root_pos=clip["root_trans_offset"],
            src_fps=float(clip.get("fps", 30.0)),
            target_fps=target_fps,
        )
        for name, clip in raw.items()
    }


def load_lafan_csv(csv_path: str | Path, src_fps: float = 30.0, target_fps: float = CONTROL_HZ) -> ReferenceMotion:
    """Unitree LAFAN1 retargeting CSV: no header, 36 columns per row at 30 fps.

    Columns: root position xyz (3), root quaternion xyzw (4), 29 joint angles in
    hardware order. Same content as the GEAR-SONIC pickle, different container.
    """
    csv_path = Path(csv_path)
    d = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != 36:
        raise ValueError(f"{csv_path}: expected (T, 36) = root pos + quat xyzw + 29 joints, got {d.shape}")
    return make_reference_motion(
        name=csv_path.stem,
        dof_hw=d[:, 7:36],
        root_quat_wxyz=quat.xyzw_to_wxyz(d[:, 3:7]),
        root_pos=d[:, 0:3],
        src_fps=src_fps,
        target_fps=target_fps,
    )


def load_motions(path: str | Path, target_fps: float = CONTROL_HZ) -> dict[str, ReferenceMotion]:
    """Load any supported motion file (``.pkl`` GEAR-SONIC, ``.csv`` LAFAN) as ``{name: clip}``."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        m = load_lafan_csv(path, target_fps=target_fps)
        return {m.name: m}
    return load_reference_motions(path, target_fps=target_fps)
