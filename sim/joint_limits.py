from __future__ import annotations

"""
Joint-limit configuration for the left arm, in **radians**.

The convention here follows a 5-DOF chain:

    0: shoulder_up_down        (flexion/extension)
    1: shoulder_left_right     (abduction/adduction)
    2: elbow_flexion
    3: upper_arm_rotation      (humerus internal/external rotation)
    4: lower_arm_rotation      (forearm pronation/supination)

The values below come from the configuration you provided:

    joint_limits:
        - [0.0, 0.52]   # Joint 1: Shoulder Up/Down   (0 to 30 degrees)
        - [0.0, 0.70]   # Joint 2: Shoulder Left/Right (0 to 40 degrees)
        - [0.0, 1.17]   # Joint 3: Elbow              (0 to 67 degrees)
        - [0.0, 1.57]   # Joint 4: Upper Arm Rotation (0 to 90 degrees)
        - [-1.57, 1.57] # Joint 5: Lower Arm Rotation (±90 degrees)

These are hardware/robot limits; the higher-level kinematics and DMP
model still operate in the same 4-DOF space:

    [elbow_flexion, shoulder_flexion, shoulder_abduction, shoulder_internal_rotation]

We map that 4-tuple into this 5-DOF limit vector as:

    elbow_flexion           -> index 2
    shoulder_flexion        -> index 0
    shoulder_abduction      -> index 1
    shoulder_internal_rot   -> index 3
    lower_arm_rotation      -> index 4 (currently unused)
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Update joint limits
JOINT_LIMITS_RAD: np.ndarray = np.array(
    [
        [0.0, 1.05],     # 0: Elbow flexion (0 to 60 degrees)
        [0.0, 1.39],     # 1: Shoulder flexion (0 to 80 degrees)
        [0.0, 0.69],     # 2: Shoulder abduction (0 to 40 degrees)
        [-0.69, 0.69],   # 3: Shoulder lateral/medial rotation (-40 to 40 degrees)
    ],
    dtype=float,
)


@dataclass(frozen=True)
class DMPToRobotIndexMapping:
    """
    Indices that map the 4-DOF DMP vector
    [elbow, shoulder_flexion, shoulder_abduction, shoulder_internal_rotation]
    onto the JOINT_LIMITS_RAD rows.
    """

    elbow: int = 0
    shoulder_flex: int = 1
    shoulder_abd: int = 2
    shoulder_int_rot: int = 3


DMP_LIMIT_INDEX = DMPToRobotIndexMapping()


def clamp_angle(angle: float, limits: Tuple[float, float]) -> float:
    """Clamp a scalar angle (radians) to [min, max]."""
    lo, hi = limits
    return float(np.clip(angle, lo, hi))


def clamp_dmp_vector(q_rad: np.ndarray) -> np.ndarray:
    """
    Clamp a 4-DOF DMP configuration (radians) to the robot joint limits.

    Args:
        q_rad: shape (..., 4) = [elbow, shoulder_flex, shoulder_abd, shoulder_int_rot]

    Returns:
        Clamped copy with the same shape.
    """
    q = np.asarray(q_rad, dtype=float).copy()
    if q.shape[-1] != 4:
        raise ValueError(f"Expected last dimension 4 for DMP vector, got shape {q.shape}")

    idx = DMP_LIMIT_INDEX
    # Build a view for the last axis
    # [..., 0] elbow
    q_elbow = q[..., 0]
    q_sh_flex = q[..., 1]
    q_sh_abd = q[..., 2]
    q_sh_int = q[..., 3]

    # np.clip sets the value to the min or max of the range: min(max(x, min), max)

    q_elbow[...] = np.clip(q_elbow, JOINT_LIMITS_RAD[idx.elbow, 0], JOINT_LIMITS_RAD[idx.elbow, 1])
    q_sh_flex[...] = np.clip(
        q_sh_flex, JOINT_LIMITS_RAD[idx.shoulder_flex, 0], JOINT_LIMITS_RAD[idx.shoulder_flex, 1]
    )
    q_sh_abd[...] = np.clip(
        q_sh_abd, JOINT_LIMITS_RAD[idx.shoulder_abd, 0], JOINT_LIMITS_RAD[idx.shoulder_abd, 1]
    )
    q_sh_int[...] = np.clip(
        q_sh_int,
        JOINT_LIMITS_RAD[idx.shoulder_int_rot, 0],
        JOINT_LIMITS_RAD[idx.shoulder_int_rot, 1],
    )

    return q

