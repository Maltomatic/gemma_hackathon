"""SONIC whole-body controller for the G1, running the released ONNX encoder/decoder on CPU.

This is a Python port of the observation pipeline in NVIDIA's C++ deploy stack
(``gear_sonic_deploy/.../g1_deploy_onnx_ref.cpp``), restricted to the ``g1`` encoder mode
(track a reference joint-space motion). Per control tick (50 Hz):

1. push the current proprioceptive state (and the previous action) into 10-frame histories;
2. build the encoder input from the next 10 reference frames (100 ms apart) and run the
   encoder to get a 64-d motion token;
3. build the decoder input (token + histories) and run the decoder to get a 29-d action
   in Isaac Lab joint order;
4. ``q_target = default_angles + action * action_scale`` in hardware order, which
   ``G1Sim`` turns into torque with SONIC's own PD gains.

Usage::

    sim = G1Sim(mode="torque", gains=SonicController.pd_gains())
    ctrl = SonicController("models/sonic/sonic_v1_1", motion=clip)
    obs = sonic_reset(sim, ctrl)
    while ...: obs = sim.step(ctrl)
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from . import quat
from .controllers import PDGains
from .env import G1Obs, G1Sim
from .sonic_motion import ReferenceMotion
from .sonic_params import (
    ACTION_CLIP, ACTION_SCALE_HW, DEC_OFF_ACTION, DEC_OFF_ANGVEL, DEC_OFF_GRAVITY, DEC_OFF_QPOS,
    DEC_OFF_QVEL, DEC_OFF_TOKEN, DECODER_INPUT_DIM, DEFAULT_ANGLES_HW, DEFAULT_ANGLES_ISAAC,
    ENC_OFF_MODE, ENC_OFF_REF_ORI, ENC_OFF_REF_QPOS, ENC_OFF_REF_QVEL, ENCODER_INPUT_DIM,
    ENCODER_MODE_G1, HISTORY_FRAMES, ISAACLAB_TO_MUJOCO, KD_HW, KP_HW, MUJOCO_TO_ISAACLAB,
    NUM_JOINTS, REF_FRAMES, REF_STEP, TOKEN_DIM, V1_1_GAIN_SCALE_HW,
)

GRAVITY_DOWN = np.array([0.0, 0.0, -1.0])


class SonicController:
    def __init__(
        self,
        model_dir: str | Path,
        motion: Optional[ReferenceMotion] = None,
        ankle_gain_boost: bool = True,
        loop: bool = False,
        providers: Optional[list[str]] = None,
    ):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        providers = providers or ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(str(model_dir / "model_encoder.onnx"), opts, providers=providers)
        self.decoder = ort.InferenceSession(str(model_dir / "model_decoder.onnx"), opts, providers=providers)
        self._check_io()

        self.motion = motion
        self.loop = loop
        self.playing = False
        self.frame = 0
        self.gain_scale_hw = V1_1_GAIN_SCALE_HW if ankle_gain_boost else np.ones(NUM_JOINTS)

        self.last_action = np.zeros(NUM_JOINTS)             # raw policy output, Isaac Lab order
        self.last_target_hw = DEFAULT_ANGLES_HW.copy()
        self.token = np.zeros(TOKEN_DIM, dtype=np.float32)
        self._hist_angvel: deque[np.ndarray] = deque(maxlen=HISTORY_FRAMES)
        self._hist_qpos: deque[np.ndarray] = deque(maxlen=HISTORY_FRAMES)
        self._hist_qvel: deque[np.ndarray] = deque(maxlen=HISTORY_FRAMES)
        self._hist_action: deque[np.ndarray] = deque(maxlen=HISTORY_FRAMES)
        self._hist_gravity: deque[np.ndarray] = deque(maxlen=HISTORY_FRAMES)
        self._delta_heading = np.array([1.0, 0.0, 0.0, 0.0])
        self._enc_in = np.zeros(ENCODER_INPUT_DIM, dtype=np.float32)
        self._dec_in = np.zeros(DECODER_INPUT_DIM, dtype=np.float32)

    # ------------------------------------------------------------------ setup
    def _check_io(self) -> None:
        e_in, = self.encoder.get_inputs()
        d_in, = self.decoder.get_inputs()
        e_out, = self.encoder.get_outputs()
        d_out, = self.decoder.get_outputs()
        assert e_in.name == "obs_dict" and d_in.name == "obs_dict", (e_in.name, d_in.name)
        assert e_out.name == "encoded_tokens" and d_out.name == "action", (e_out.name, d_out.name)
        assert int(np.prod(e_in.shape)) == ENCODER_INPUT_DIM, e_in.shape
        assert int(np.prod(d_in.shape)) == DECODER_INPUT_DIM, d_in.shape
        assert int(np.prod(e_out.shape)) == TOKEN_DIM, e_out.shape
        assert int(np.prod(d_out.shape)) == NUM_JOINTS, d_out.shape
        self._enc_shape = [int(s) if isinstance(s, int) else 1 for s in e_in.shape]
        self._dec_shape = [int(s) if isinstance(s, int) else 1 for s in d_in.shape]

    @staticmethod
    def pd_gains(ankle_gain_boost: bool = True) -> PDGains:
        """SONIC's motor-level PD gains in hardware order, for ``G1Sim(gains=...)``."""
        s = V1_1_GAIN_SCALE_HW if ankle_gain_boost else np.ones(NUM_JOINTS)
        return PDGains(kp=KP_HW * s, kd=KD_HW * s)

    def set_motion(self, motion: ReferenceMotion) -> None:
        self.motion = motion
        self.frame = 0
        self.playing = False

    def play(self, playing: bool = True) -> None:
        self.playing = playing

    def restart(self) -> None:
        self.frame = 0

    # ------------------------------------------------------------------ state
    def reset(self, obs: Optional[G1Obs] = None) -> None:
        """Clear histories and re-anchor the reference heading to the robot's current heading.

        Mirrors the C++ ``HeadingState`` init: the reference clip is yaw-rotated so its
        first frame faces the way the robot faces when control starts.
        """
        self.frame = 0
        self.playing = False
        self.last_action = np.zeros(NUM_JOINTS)
        self.last_target_hw = DEFAULT_ANGLES_HW.copy()
        for h in (self._hist_angvel, self._hist_qpos, self._hist_qvel, self._hist_action, self._hist_gravity):
            h.clear()
        if obs is not None:
            if self.motion is not None:
                init_heading = quat.heading_quat(obs.base_quat)
                data_heading_inv = quat.heading_quat_inv(self.motion.root_quat[0])
                self._delta_heading = quat.mul(init_heading, data_heading_inv)
            # Isaac Lab history buffers are filled with the first observation on reset.
            for _ in range(HISTORY_FRAMES):
                self._push_state(obs)

    def _push_state(self, obs: G1Obs) -> None:
        self._hist_angvel.append(obs.base_ang_vel.astype(np.float64))
        self._hist_qpos.append(obs.q[MUJOCO_TO_ISAACLAB] - DEFAULT_ANGLES_ISAAC)
        self._hist_qvel.append(obs.dq[MUJOCO_TO_ISAACLAB])
        self._hist_action.append(self.last_action.copy())
        self._hist_gravity.append(quat.rotate(quat.conj(obs.base_quat), GRAVITY_DOWN))

    # ------------------------------------------------------------ observations
    def _ref_frame(self, k: int) -> int:
        """Reference frame index for lookahead slot k (held at the cursor when not playing)."""
        f = self.frame + (k * REF_STEP if self.playing else 0)
        return self.motion.clamp(f)

    def build_encoder_input(self, obs: G1Obs) -> np.ndarray:
        m = self.motion
        x = self._enc_in
        x[:] = 0.0
        x[ENC_OFF_MODE] = ENCODER_MODE_G1
        robot_heading = quat.heading_quat(obs.base_quat)
        robot_heading_inv = quat.conj(robot_heading)
        for k in range(REF_FRAMES):
            f = self._ref_frame(k)
            x[ENC_OFF_REF_QPOS + k * NUM_JOINTS: ENC_OFF_REF_QPOS + (k + 1) * NUM_JOINTS] = m.qpos_isaac[f]
            if self.playing:
                x[ENC_OFF_REF_QVEL + k * NUM_JOINTS: ENC_OFF_REF_QVEL + (k + 1) * NUM_JOINTS] = m.qvel_isaac[f]
            ref_rot = quat.mul(self._delta_heading, m.root_quat[f])
            x[ENC_OFF_REF_ORI + k * 6: ENC_OFF_REF_ORI + (k + 1) * 6] = quat.to_6d(quat.mul(robot_heading_inv, ref_rot))
        return x

    def build_decoder_input(self) -> np.ndarray:
        x = self._dec_in
        x[DEC_OFF_TOKEN:DEC_OFF_TOKEN + TOKEN_DIM] = self.token
        x[DEC_OFF_ANGVEL:DEC_OFF_ANGVEL + 30] = np.concatenate(self._hist_angvel)
        x[DEC_OFF_QPOS:DEC_OFF_QPOS + 290] = np.concatenate(self._hist_qpos)
        x[DEC_OFF_QVEL:DEC_OFF_QVEL + 290] = np.concatenate(self._hist_qvel)
        x[DEC_OFF_ACTION:DEC_OFF_ACTION + 290] = np.concatenate(self._hist_action)
        x[DEC_OFF_GRAVITY:DEC_OFF_GRAVITY + 30] = np.concatenate(self._hist_gravity)
        return x

    # ---------------------------------------------------------------- control
    def __call__(self, obs: G1Obs) -> np.ndarray:
        if self.motion is None:
            raise RuntimeError("SonicController needs a reference motion (set_motion) in g1 mode")
        if len(self._hist_qpos) == 0:
            self.reset(obs)
        else:
            self._push_state(obs)

        enc_in = self.build_encoder_input(obs).reshape(self._enc_shape)
        self.token = self.encoder.run(None, {"obs_dict": enc_in})[0].reshape(-1).astype(np.float32)

        dec_in = self.build_decoder_input().reshape(self._dec_shape)
        action = self.decoder.run(None, {"obs_dict": dec_in})[0].reshape(-1).astype(np.float64)
        action = np.clip(action, -ACTION_CLIP, ACTION_CLIP)
        self.last_action = action

        target_hw = DEFAULT_ANGLES_HW + action[ISAACLAB_TO_MUJOCO] * ACTION_SCALE_HW
        self.last_target_hw = target_hw

        if self.playing:
            self.frame += 1
            if self.frame >= self.motion.num_frames:
                self.frame = 0 if self.loop else self.motion.num_frames - 1
        return target_hw

    # -------------------------------------------------------------- metrics
    def tracking_error(self, obs: G1Obs) -> np.ndarray:
        """|q - q_ref| per joint (hardware order) at the current reference cursor."""
        f = self.motion.clamp(self.frame)
        return np.abs(obs.q - self.motion.qpos_hw[f])


def settle_default_pose(sim: G1Sim, seconds: float = 0.6, drop_height: float = 0.76) -> G1Obs:
    """Put the G1 in SONIC's default standing pose and let it settle onto the floor.

    Stiff PD on the joints during the drop so the pose survives the landing; the caller
    then switches to the SONIC gains by constructing the sim with them (apply_targets uses
    sim.gains, so we temporarily swap).
    """
    import mujoco

    mujoco.mj_resetData(sim.model, sim.data)
    sim.data.qpos[:] = 0.0
    sim.data.qpos[2] = drop_height
    sim.data.qpos[3] = 1.0
    sim.data.qpos[sim.qpos_adr] = DEFAULT_ANGLES_HW
    mujoco.mj_forward(sim.model, sim.data)

    saved_gains, saved_mode = sim.gains, sim.mode
    sim.gains = PDGains(kp=np.full(NUM_JOINTS, 300.0), kd=np.full(NUM_JOINTS, 10.0))
    sim.mode = "torque"
    try:
        for _ in range(int(seconds / sim.sim_dt)):
            sim.substep(DEFAULT_ANGLES_HW)
    finally:
        sim.gains, sim.mode = saved_gains, saved_mode
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)
    sim.last_target_q = DEFAULT_ANGLES_HW.copy()
    return sim.observe()


def sonic_reset(sim: G1Sim, ctrl: SonicController) -> G1Obs:
    obs = settle_default_pose(sim)
    ctrl.reset(obs)
    return obs
