"""MuJoCo simulation of the Unitree G1 (29-DoF) with a pluggable controller hook."""
from .env import G1Sim, G1Obs, ACTUATED_JOINTS, MENAGERIE_G1_DIR
from .controllers import Controller, HoldPoseController, PDGains, DEFAULT_RL_GAINS, TORQUE_LIMITS

__all__ = [
    "G1Sim", "G1Obs", "ACTUATED_JOINTS", "MENAGERIE_G1_DIR",
    "Controller", "HoldPoseController", "PDGains", "DEFAULT_RL_GAINS", "TORQUE_LIMITS",
]

# SONIC (optional: needs onnxruntime + joblib and the downloaded checkpoint)
try:
    from .sonic_controller import SonicController, sonic_reset, settle_default_pose  # noqa: F401
    from .sonic_motion import ReferenceMotion, load_reference_motions  # noqa: F401
    __all__ += ["SonicController", "sonic_reset", "settle_default_pose", "ReferenceMotion", "load_reference_motions"]
except ImportError:  # pragma: no cover
    pass
