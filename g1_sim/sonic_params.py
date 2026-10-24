"""SONIC deployment constants for the Unitree G1 29-DoF.

Everything here is transcribed from NVIDIA's GR00T-WholeBodyControl deploy stack
(gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp and
gear_sonic/envs/manager_env/robots/g1.py), so the Python port produces the same
tensors the C++ TensorRT runtime feeds the ONNX models.

Two joint orderings coexist:

* **hardware / MuJoCo order** (``g1_sim.controllers.ACTUATED_JOINTS``): legs L/R,
  waist, left arm, right arm. Used by the Unitree SDK, the Menagerie MJCF, and
  the motor command.
* **Isaac Lab order**: breadth-first over the kinematic tree, which interleaves
  left/right (left_hip_pitch, right_hip_pitch, waist_yaw, left_hip_roll, ...).
  Used by the policy observation and action vectors and by reference motions.
"""
from __future__ import annotations

import numpy as np

NUM_JOINTS = 29

# ---------------------------------------------------------------- joint order
# Isaac Lab joint order, by name (hip_pitch_link -> hip_pitch_joint etc.).
ISAACLAB_JOINTS: tuple[str, ...] = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)

# MUJOCO_TO_ISAACLAB[i] = hardware index of Isaac Lab joint i  (x_isaac = x_hw[MUJOCO_TO_ISAACLAB])
MUJOCO_TO_ISAACLAB = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
])
# ISAACLAB_TO_MUJOCO[i] = Isaac Lab index of hardware joint i  (x_hw = x_isaac[ISAACLAB_TO_MUJOCO])
ISAACLAB_TO_MUJOCO = np.array([
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
])

# ------------------------------------------------------------- motor physics
# Per-motor armature; gains come from a 10 Hz critically-damped (zeta = 2) model.
_ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
_EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
_W = 10.0 * 2.0 * np.pi
_ZETA = 2.0
_STIFF = {k: a * _W * _W for k, a in _ARMATURE.items()}
_DAMP = {k: 2.0 * _ZETA * a * _W for k, a in _ARMATURE.items()}

# (motor type, gain multiplier) per joint in hardware order.
_MOTOR_SPEC: tuple[tuple[str, float], ...] = (
    ("7520_22", 1.0), ("7520_22", 1.0), ("7520_14", 1.0), ("7520_22", 1.0), ("5020", 2.0), ("5020", 2.0),  # left leg
    ("7520_22", 1.0), ("7520_22", 1.0), ("7520_14", 1.0), ("7520_22", 1.0), ("5020", 2.0), ("5020", 2.0),  # right leg
    ("7520_14", 1.0), ("5020", 2.0), ("5020", 2.0),                                                       # waist
    ("5020", 1.0), ("5020", 1.0), ("5020", 1.0), ("5020", 1.0), ("5020", 1.0), ("4010", 1.0), ("4010", 1.0),  # left arm
    ("5020", 1.0), ("5020", 1.0), ("5020", 1.0), ("5020", 1.0), ("5020", 1.0), ("4010", 1.0), ("4010", 1.0),  # right arm
)

KP_HW = np.array([_STIFF[m] * g for m, g in _MOTOR_SPEC])
KD_HW = np.array([_DAMP[m] * g for m, g in _MOTOR_SPEC])
# action_scale = 0.25 * effort_limit / stiffness (gain multiplier NOT applied, matches the C++ tables)
ACTION_SCALE_HW = np.array([0.25 * _EFFORT[m] / _STIFF[m] for m, _ in _MOTOR_SPEC])

# Default (standing) joint angles, hardware order. Actions are offsets from these.
DEFAULT_ANGLES_HW = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
])
DEFAULT_ANGLES_ISAAC = DEFAULT_ANGLES_HW[MUJOCO_TO_ISAACLAB]

# SONIC v1.1 tested deployment tuning: ankle-pitch motors (hardware 4 and 10) x1.5 on kp and kd.
V1_1_GAIN_SCALE_HW = np.ones(NUM_JOINTS)
V1_1_GAIN_SCALE_HW[[4, 10]] = 1.5

# ------------------------------------------------------------------ policy
CONTROL_HZ = 50
HISTORY_FRAMES = 10          # proprioceptive history, step 1 (consecutive 20 ms ticks)
REF_FRAMES = 10              # reference lookahead frames
REF_STEP = 5                 # ticks between reference frames (100 ms at 50 Hz)
TOKEN_DIM = 64
ACTION_CLIP = 20.0           # env_config.action_clip_value

ENCODER_INPUT_DIM = 1751
DECODER_INPUT_DIM = 994
ENCODER_MODE_G1 = 0

# Encoder layout (sonic_v1_1/observation_config.yaml), offsets into the 1751 vector.
# Only the g1-mode terms are filled; everything else stays zero.
ENC_OFF_MODE = 0                                   # encoder_mode_4            (4)
ENC_OFF_REF_QPOS = 4                               # motion_joint_positions_10frame_step5   (290)
ENC_OFF_REF_QVEL = ENC_OFF_REF_QPOS + 290          # motion_joint_velocities_10frame_step5  (290)
ENC_OFF_REF_ORI = ENC_OFF_REF_QVEL + 290           # motion_anchor_orientation_heading_10frame_step5 (60)
ENC_G1_USED = ENC_OFF_REF_ORI + 60                 # = 644; remaining 1107 are teleop / smpl terms

# Decoder layout (sonic_v1_1/observation_config.yaml): token, then 10-frame histories.
DEC_OFF_TOKEN = 0
DEC_OFF_ANGVEL = 64
DEC_OFF_QPOS = DEC_OFF_ANGVEL + 10 * 3
DEC_OFF_QVEL = DEC_OFF_QPOS + 10 * 29
DEC_OFF_ACTION = DEC_OFF_QVEL + 10 * 29
DEC_OFF_GRAVITY = DEC_OFF_ACTION + 10 * 29
assert DEC_OFF_GRAVITY + 30 == DECODER_INPUT_DIM
