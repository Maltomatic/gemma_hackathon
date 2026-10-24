"""G1Sim: thin wrapper around MuJoCo for the Unitree G1 29-DoF model.

Design goals
------------
* One place that owns model/data and the control loop.
* ``step(controller)`` runs one *policy* tick (default 50 Hz) which is
  ``decimation`` physics substeps (default 10 x 2 ms).
* Two ways of turning target joint positions into torque:
  - ``mode="position"``: MuJoCo's built-in position actuators (kp=500 from the
    Menagerie MJCF). Stiff and stable; fine for sanity checks.
  - ``mode="torque"``: actuators are converted to pure torque motors at load
    time and an explicit PD loop runs in Python at the physics rate. This is
    what a sim-to-real policy like SONIC expects, because the gains then match
    the real robot's motor-level PD.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import mujoco
import numpy as np

from .controllers import (
    ACTUATED_JOINTS, NUM_ACTUATORS, Controller, DEFAULT_RL_GAINS, PDGains, TORQUE_LIMITS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MENAGERIE_G1_DIR = REPO_ROOT / "third_party" / "mujoco_menagerie" / "unitree_g1"
DEFAULT_SCENE = MENAGERIE_G1_DIR / "scene.xml"

__all__ = ["G1Sim", "G1Obs", "ACTUATED_JOINTS", "MENAGERIE_G1_DIR", "DEFAULT_SCENE"]


@dataclass
class G1Obs:
    """Proprioceptive observation, everything a blind whole-body policy needs."""

    time: float
    q: np.ndarray             # (29,) joint positions, actuator order
    dq: np.ndarray            # (29,) joint velocities
    base_pos: np.ndarray      # (3,) pelvis position in world (sim only, not on robot)
    base_quat: np.ndarray     # (4,) pelvis orientation, MuJoCo order (w, x, y, z)
    base_lin_vel: np.ndarray  # (3,) pelvis linear velocity, world frame (sim only)
    base_ang_vel: np.ndarray  # (3,) pelvis angular velocity, body frame
    proj_gravity: np.ndarray  # (3,) gravity direction in pelvis frame
    imu_gyro: np.ndarray      # (3,) pelvis IMU gyro sensor (noisy, like hardware)
    imu_accel: np.ndarray     # (3,) pelvis IMU accelerometer sensor
    last_target_q: np.ndarray # (29,) previous commanded target


class G1Sim:
    def __init__(
        self,
        scene_xml: Path | str = DEFAULT_SCENE,
        mode: Literal["position", "torque"] = "torque",
        gains: PDGains = DEFAULT_RL_GAINS,
        control_hz: float = 50.0,
        torque_limits: Optional[np.ndarray] = TORQUE_LIMITS,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)
        self.mode = mode
        self.gains = gains
        self.torque_limits = None if torque_limits is None else np.asarray(torque_limits)

        self.sim_dt = float(self.model.opt.timestep)
        self.decimation = int(round(1.0 / (control_hz * self.sim_dt)))
        self.control_dt = self.decimation * self.sim_dt

        # Resolve indices once.
        self.act_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATED_JOINTS])
        assert (self.act_ids >= 0).all(), "actuator names changed in MJCF"
        jnt_ids = [self.model.actuator_trnid[a, 0] for a in self.act_ids]
        self.qpos_adr = np.array([self.model.jnt_qposadr[j] for j in jnt_ids])
        self.qvel_adr = np.array([self.model.jnt_dofadr[j] for j in jnt_ids])
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._gyro_adr = self._sensor_adr("imu-pelvis-angular-velocity")
        self._accel_adr = self._sensor_adr("imu-pelvis-linear-acceleration")
        self.key_stand = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")

        if mode == "torque":
            self._convert_actuators_to_motors()

        self.last_target_q = np.zeros(NUM_ACTUATORS)
        self.reset()

    # ----------------------------------------------------------------- setup
    def _sensor_adr(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        assert sid >= 0, name
        return int(self.model.sensor_adr[sid])

    def _convert_actuators_to_motors(self) -> None:
        """Turn the MJCF <position> actuators into direct torque motors.

        MuJoCo's general actuator computes force = gain*ctrl + bias0 + bias1*q + bias2*dq.
        A <position kp kv> is gain=kp, bias=[0,-kp,-kv]. Setting gain=1, bias=0 gives
        force = ctrl, i.e. a plain motor. ctrlrange becomes the torque limit.
        """
        m = self.model
        m.actuator_gainprm[:, :] = 0.0
        m.actuator_gainprm[:, 0] = 1.0
        m.actuator_biasprm[:, :] = 0.0
        m.actuator_dyntype[:] = mujoco.mjtDyn.mjDYN_NONE
        if self.torque_limits is not None:
            m.actuator_ctrllimited[:] = 1
            m.actuator_ctrlrange[self.act_ids, 0] = -self.torque_limits
            m.actuator_ctrlrange[self.act_ids, 1] = self.torque_limits
        else:
            m.actuator_ctrllimited[:] = 0

    # ----------------------------------------------------------------- state
    def reset(self, keyframe: Optional[str] = "stand") -> G1Obs:
        mujoco.mj_resetData(self.model, self.data)
        if keyframe is not None:
            kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            assert kid >= 0, f"no keyframe {keyframe!r}"
            mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        mujoco.mj_forward(self.model, self.data)
        self.last_target_q = self.q.copy()
        return self.observe()

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr]

    @property
    def dq(self) -> np.ndarray:
        return self.data.qvel[self.qvel_adr]

    @property
    def stand_q(self) -> np.ndarray:
        """Joint targets from the ``stand`` keyframe, actuator order."""
        return self.model.key_qpos[self.key_stand][self.qpos_adr].copy()

    def observe(self) -> G1Obs:
        d = self.data
        quat = d.xquat[self.pelvis_id].copy()  # (w, x, y, z)
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        proj_gravity = rot.T @ np.array([0.0, 0.0, -1.0])  # world gravity dir -> pelvis frame
        return G1Obs(
            time=float(d.time),
            q=self.q.copy(),
            dq=self.dq.copy(),
            base_pos=d.xpos[self.pelvis_id].copy(),
            base_quat=quat,
            base_lin_vel=d.qvel[0:3].copy(),
            base_ang_vel=d.qvel[3:6].copy(),  # free-joint angular qvel is body frame
            proj_gravity=proj_gravity,
            imu_gyro=d.sensordata[self._gyro_adr:self._gyro_adr + 3].copy(),
            imu_accel=d.sensordata[self._accel_adr:self._accel_adr + 3].copy(),
            last_target_q=self.last_target_q.copy(),
        )

    # --------------------------------------------------------------- control
    def apply_targets(self, target_q: np.ndarray) -> None:
        """Write one physics-substep worth of control for the given joint targets."""
        target_q = np.asarray(target_q, dtype=np.float64)
        if self.mode == "position":
            self.data.ctrl[self.act_ids] = target_q
        else:
            tau = self.gains.kp * (target_q - self.q) - self.gains.kd * self.dq
            if self.torque_limits is not None:
                tau = np.clip(tau, -self.torque_limits, self.torque_limits)
            self.data.ctrl[self.act_ids] = tau

    def step(self, controller: Controller) -> G1Obs:
        """One policy tick: query the controller once, run ``decimation`` substeps."""
        obs = self.observe()
        target_q = np.asarray(controller(obs), dtype=np.float64)
        assert target_q.shape == (NUM_ACTUATORS,), target_q.shape
        for _ in range(self.decimation):
            self.apply_targets(target_q)  # PD re-evaluated each substep with fresh q, dq
            mujoco.mj_step(self.model, self.data)
        self.last_target_q = target_q.copy()
        return self.observe()

    def substep(self, target_q: np.ndarray) -> None:
        """Single physics step with fixed targets (for viewer loops that pace themselves)."""
        self.apply_targets(target_q)
        mujoco.mj_step(self.model, self.data)
