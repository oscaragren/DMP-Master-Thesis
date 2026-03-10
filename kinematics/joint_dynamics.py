from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter


JOINT_NAMES_4DOF = [
    "elbow_flexion",
    "shoulder_elevation",
    "shoulder_azimuth",
    "shoulder_internal_rotation",
]


@dataclass(frozen=True)
class JointLimitsDeg:
    """Per‑joint kinematic limits in degrees / deg/s / deg/s^2."""

    pos_min: np.ndarray  # shape (n_joints,)
    pos_max: np.ndarray  # shape (n_joints,)
    vel_max_abs: np.ndarray  # shape (n_joints,)
    acc_max_abs: np.ndarray  # shape (n_joints,)

    @property
    def n_joints(self) -> int:
        return int(self.pos_min.shape[0])


def default_human_arm_limits_4dof() -> JointLimitsDeg:
    """Reasonable, conservative limits for 4‑DOF human arm model (degrees).

    Order matches JOINT_NAMES_4DOF.
    These can be tuned later if you calibrate per‑subject limits.
    """

    # Position ranges (deg)
    # Elbow flexion:       0   – 150  (0 = extended, 150 = flexed)
    # Shoulder elevation:  0   – 180  (0 = arm up, 180 = arm down)
    # Shoulder azimuth:   -150 – 150  (across body to abducted)
    # Shoulder int. rot.: -120 – 120
    pos_min = np.array([0.0, 0.0, -150.0, -120.0], dtype=float)
    pos_max = np.array([150.0, 180.0, 150.0, 120.0], dtype=float)

    # Velocity limits (deg/s), fairly generous for fast but plausible motion.
    vel_max_abs = np.array([450.0, 360.0, 360.0, 360.0], dtype=float)

    # Acceleration limits (deg/s^2), again conservative but not too tight.
    acc_max_abs = np.array([4000.0, 3500.0, 3500.0, 3500.0], dtype=float)

    return JointLimitsDeg(
        pos_min=pos_min,
        pos_max=pos_max,
        vel_max_abs=vel_max_abs,
        acc_max_abs=acc_max_abs,
    )


def smooth_angles_deg(
    q_deg: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Smooth demonstrated joint angles (deg) along time using Savitzky–Golay.

    Args:
        q_deg: (T, n_joints) angles in degrees.
        window_length: Odd filter length in samples (will be adapted if too long).
        polyorder: Polynomial order for Savitzky–Golay.
    """
    q = np.asarray(q_deg, dtype=float)
    if q.ndim != 2:
        raise ValueError(f"Expected (T, n_joints) array, got shape {q.shape}")

    T, n_joints = q.shape
    if T < 3:
        return q.copy()

    # Ensure odd window length and not larger than T.
    wl = min(window_length, T if T % 2 == 1 else T - 1)
    if wl < 3:
        wl = 3 if T >= 3 else T | 1

    wl = max(3, wl)
    if wl <= polyorder:
        polyorder = max(1, wl - 1)

    q_smooth = np.empty_like(q)
    for j in range(n_joints):
        q_smooth[:, j] = savgol_filter(q[:, j], wl, polyorder, mode="interp")
    return q_smooth


def finite_differences(
    q: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute velocity and acceleration by finite differences.

    Args:
        q: (T, n_joints) positions.
        dt: Timestep in seconds.

    Returns:
        dq: (T, n_joints) velocities.
        ddq: (T, n_joints) accelerations.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 2:
        raise ValueError(f"Expected (T, n_joints) array, got shape {q.shape}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    dq = np.gradient(q, dt, axis=0)
    ddq = np.gradient(dq, dt, axis=0)
    return dq, ddq


@dataclass
class TrajectoryValidationReport:
    ok: bool
    reason: str
    position_violations: int
    velocity_violations: int
    acceleration_violations: int
    has_nans: bool


def validate_joint_trajectory_deg(
    q_deg: np.ndarray,
    dt: float,
    limits: Optional[JointLimitsDeg] = None,
    raise_on_error: bool = False,
    name: str = "trajectory",
) -> TrajectoryValidationReport:
    """Validate a joint‑space trajectory in degrees against simple human limits.

    Checks:
        - finite values
        - position inside per‑joint [min, max]
        - |velocity| <= vel_max_abs
        - |acceleration| <= acc_max_abs
    """
    q = np.asarray(q_deg, dtype=float)
    if q.ndim != 2:
        raise ValueError(f"Expected (T, n_joints) array, got shape {q.shape}")
    T, n_joints = q.shape

    if limits is None:
        limits = default_human_arm_limits_4dof()

    if n_joints != limits.n_joints:
        raise ValueError(
            f"{name}: expected {limits.n_joints} joints for validation, got {n_joints}"
        )

    has_nans = not np.all(np.isfinite(q))

    pos_min = limits.pos_min[None, :]
    pos_max = limits.pos_max[None, :]

    below = q < pos_min
    above = q > pos_max
    position_violations = int(np.count_nonzero(below | above))

    dq, ddq = finite_differences(q, dt)
    vel_max = limits.vel_max_abs[None, :]
    acc_max = limits.acc_max_abs[None, :]

    vel_viol = np.abs(dq) > vel_max
    acc_viol = np.abs(ddq) > acc_max

    velocity_violations = int(np.count_nonzero(vel_viol))
    acceleration_violations = int(np.count_nonzero(acc_viol))

    ok = (
        (not has_nans)
        and position_violations == 0
        and velocity_violations == 0
        and acceleration_violations == 0
    )

    if ok:
        reason = f"{name}: OK (within joint limits)"
    else:
        parts = []
        if has_nans:
            parts.append("NaNs/Infs present")
        if position_violations:
            parts.append(f"{position_violations} position violations")
        if velocity_violations:
            parts.append(f"{velocity_violations} velocity violations")
        if acceleration_violations:
            parts.append(f"{acceleration_violations} acceleration violations")
        reason = f"{name}: " + ", ".join(parts)

    report = TrajectoryValidationReport(
        ok=ok,
        reason=reason,
        position_violations=position_violations,
        velocity_violations=velocity_violations,
        acceleration_violations=acceleration_violations,
        has_nans=has_nans,
    )

    if raise_on_error and not ok:
        raise ValueError(reason)

    return report

