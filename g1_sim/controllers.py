"""Controller interface and baseline controllers for the G1.

A controller maps an observation to a 29-vector of *target joint positions*
(radians) in the actuator order given by ``g1_sim.env.ACTUATED_JOINTS``.
The sim then turns those targets into torques, either with MuJoCo's built-in
position actuators or with an explicit PD loop (see ``G1Sim(mode=...)``).

SONIC (or any learned policy) should be wrapped as a ``Controller`` so the
sim code never has to change: observe -> policy -> target q -> PD -> torque.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .env import G1Obs


# Actuator order of the Menagerie g1.xml. This is also the joint order the
# Unitree SDK and most G1 RL policies (incl. SONIC's 29-DoF setup) use:
# legs (L then R), waist, left arm, right arm.
ACTUATED_JOINTS: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
NUM_ACTUATORS = len(ACTUATED_JOINTS)


class Controller(Protocol):
    """Anything with ``__call__(obs) -> target_q[29]`` is a controller."""

    def reset(self) -> None: ...

    def __call__(self, obs: "G1Obs") -> np.ndarray: ...


class HoldPoseController:
    """Returns a fixed target pose every tick. Default: the model's ``stand`` keyframe.

    This is the "zero policy": it proves the PD loop and the sim are wired
    correctly before a learned policy goes in.
    """

    def __init__(self, target_q: np.ndarray):
        self.target_q = np.asarray(target_q, dtype=np.float64).copy()
        assert self.target_q.shape == (NUM_ACTUATORS,), self.target_q.shape

    def reset(self) -> None:
        pass

    def __call__(self, obs: "G1Obs") -> np.ndarray:
        return self.target_q


@dataclass(frozen=True)
class PDGains:
    """Per-joint PD gains in actuator order. tau = kp*(q_des - q) - kd*dq."""

    kp: np.ndarray
    kd: np.ndarray

    @staticmethod
    def from_groups(kp_by_group: dict[str, float], kd_by_group: dict[str, float]) -> "PDGains":
        def group_of(name: str) -> str:
            for key in ("hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"):
                if key in name:
                    return key
            raise KeyError(name)

        kp = np.array([kp_by_group[group_of(j)] for j in ACTUATED_JOINTS], dtype=np.float64)
        kd = np.array([kd_by_group[group_of(j)] for j in ACTUATED_JOINTS], dtype=np.float64)
        return PDGains(kp=kp, kd=kd)

    @staticmethod
    def from_joint_dicts(kp: dict[str, float], kd: dict[str, float]) -> "PDGains":
        """Build from per-joint dicts keyed by joint name (how most RL deploy configs store gains)."""
        return PDGains(
            kp=np.array([kp[j] for j in ACTUATED_JOINTS], dtype=np.float64),
            kd=np.array([kd[j] for j in ACTUATED_JOINTS], dtype=np.float64),
        )


# Gains in the range used by common G1 RL deployments (unitree_rl_gym style).
# PLACEHOLDER: SONIC ships its own per-joint kp/kd in its deployment config.
# Load those into a PDGains and pass them to G1Sim(gains=...) when wiring it in.
DEFAULT_RL_GAINS = PDGains.from_groups(
    kp_by_group={"hip": 100.0, "knee": 150.0, "ankle": 40.0, "waist": 150.0,
                 "shoulder": 40.0, "elbow": 40.0, "wrist": 20.0},
    kd_by_group={"hip": 2.0, "knee": 4.0, "ankle": 2.0, "waist": 3.0,
                 "shoulder": 1.0, "elbow": 1.0, "wrist": 0.5},
)


def _torque_limit(name: str) -> float:
    """Peak joint torque [Nm], from Unitree's published G1 29-DoF motor specs.

    hip / waist-yaw 88, knee 139, ankle / waist-roll / waist-pitch 50,
    shoulder / elbow / wrist-roll 25, wrist-pitch / wrist-yaw 5.
    """
    if "knee" in name:
        return 139.0
    if "hip" in name or name == "waist_yaw_joint":
        return 88.0
    if "ankle" in name or "waist" in name:
        return 50.0
    if "wrist_pitch" in name or "wrist_yaw" in name:
        return 5.0
    return 25.0  # shoulder, elbow, wrist roll


TORQUE_LIMITS = np.array([_torque_limit(j) for j in ACTUATED_JOINTS], dtype=np.float64)
