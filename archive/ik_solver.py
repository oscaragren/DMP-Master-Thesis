from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def R_x(a: float) -> np.ndarray:
    ca = np.cos(a)
    sa = np.sin(a)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, ca, -sa],
            [0.0, sa, ca],
        ],
        dtype=np.float64,
    )


def R_y(a: float) -> np.ndarray:
    ca = np.cos(a)
    sa = np.sin(a)
    return np.array(
        [
            [ca, 0.0, sa],
            [0.0, 1.0, 0.0],
            [-sa, 0.0, ca],
        ],
        dtype=np.float64,
    )


def R_z(a: float) -> np.ndarray:
    ca = np.cos(a)
    sa = np.sin(a)
    return np.array(
        [
            [ca, -sa, 0.0],
            [sa, ca, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def fk(
    theta: np.ndarray,
    shoulder_pos: np.ndarray,
    lengths: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward kinematics for a simple 4-DOF shoulder+elbow model.

    Args:
        theta: (4,) radians [flex, abd, internal_rot, elbow_flex]
        shoulder_pos: (3,) world position
        lengths: (L1, L2) upper-arm, forearm lengths

    Returns:
        elbow_pos: (3,)
        wrist_pos: (3,)
    """
    L1, L2 = float(lengths[0]), float(lengths[1])
    flex, abd, rot, elbow = [float(x) for x in theta]

    # Shoulder rotation order must be consistent with residual fitting.
    R_sh = R_z(abd) @ R_x(-flex) @ R_y(rot)

    elbow_pos = shoulder_pos + R_sh @ np.array([0.0, -L1, 0.0], dtype=np.float64)

    R_el = R_sh @ R_x(elbow)
    wrist_pos = elbow_pos + R_el @ np.array([0.0, -L2, 0.0], dtype=np.float64)

    return elbow_pos, wrist_pos


def residuals(
    theta: np.ndarray,
    shoulder: np.ndarray,
    elbow_obs: np.ndarray,
    wrist_obs: np.ndarray,
    lengths: tuple[float, float],
    prev_theta: np.ndarray | None = None,
    *,
    smooth_w: float = 0.1,
    limit_w: float = 0.05,
    elbow_w: float = 1.0,
    wrist_w: float = 1.0,
) -> np.ndarray:
    elbow_pred, wrist_pred = fk(theta, shoulder, lengths)

    res: list[float] = []

    # position errors
    res.extend((elbow_w * (elbow_pred - elbow_obs)).tolist())
    res.extend((wrist_w * (wrist_pred - wrist_obs)).tolist())

    # smoothness
    if prev_theta is not None:
        res.extend((smooth_w * (theta - prev_theta)).tolist())

    # soft joint limits (penalize only violations)
    # Limits (radians), order: [shoulder_flex, shoulder_abd, shoulder_int_rot, elbow_flex]
    theta_min = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    theta_max = np.array([0.52, 0.7, 1.57, 1.17], dtype=np.float64)

    res.extend((limit_w * np.maximum(0.0, theta - theta_max)).tolist())
    res.extend((limit_w * np.maximum(0.0, theta_min - theta)).tolist())

    return np.asarray(res, dtype=np.float64)


def solve_ik_frame(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    lengths: tuple[float, float],
    theta_init: np.ndarray,
    prev_theta: np.ndarray | None,
    *,
    smooth_w: float = 0.1,
    limit_w: float = 0.05,
    elbow_w: float = 1.0,
    wrist_w: float = 1.0,
) -> np.ndarray:
    result = least_squares(
        residuals,
        x0=np.asarray(theta_init, dtype=np.float64),
        args=(shoulder, elbow, wrist, lengths, prev_theta),
        kwargs={
            "smooth_w": float(smooth_w),
            "limit_w": float(limit_w),
            "elbow_w": float(elbow_w),
            "wrist_w": float(wrist_w),
        },
        method="lm",
    )
    return np.asarray(result.x, dtype=np.float64)


def solve_ik_sequence(
    seq: np.ndarray,
    *,
    prev_angles_deg: np.ndarray | None = None,
    smooth_w: float = 0.1,
    limit_w: float = 0.05,
    elbow_w: float = 1.0,
    wrist_w: float = 1.0,
) -> np.ndarray:
    """
    Solve IK per timestep with temporal smoothing.

    Args:
        seq: (T, N, 3) with at least indices [0]=shoulder, [1]=elbow, [2]=wrist (meters).
        prev_angles_deg: optional (T, 4) degrees used as a prior for smoothing/initialization.

    Returns:
        (T, 4) degrees [shoulder_flex, shoulder_abd, shoulder_internal_rot, elbow_flex]

    Important:
        Internal rotation is not directly observed from shoulder/elbow/wrist positions alone.
        This solver returns a **physically plausible estimate** stabilized by smoothness,
        soft joint limits, and kinematic consistency.
    """
    if seq.ndim != 3 or seq.shape[2] != 3 or seq.shape[1] < 3:
        raise ValueError(f"Expected seq shape (T, N>=3, 3), got {seq.shape}")

    T = int(seq.shape[0])
    out = np.full((T, 4), np.nan, dtype=np.float64)

    prev_theta: np.ndarray | None = None
    prev_theta_prior = None
    if prev_angles_deg is not None:
        prev_angles_deg = np.asarray(prev_angles_deg, dtype=np.float64)
        if prev_angles_deg.shape != (T, 4):
            raise ValueError(
                f"prev_angles_deg must have shape (T, 4) matching seq (T={T}), got {prev_angles_deg.shape}"
            )
        prev_theta_prior = np.deg2rad(prev_angles_deg)

    for t in range(T):
        shoulder = np.asarray(seq[t, 0], dtype=np.float64)
        elbow = np.asarray(seq[t, 1], dtype=np.float64)
        wrist = np.asarray(seq[t, 2], dtype=np.float64)

        if not (
            np.all(np.isfinite(shoulder))
            and np.all(np.isfinite(elbow))
            and np.all(np.isfinite(wrist))
        ):
            prev_theta = prev_theta if prev_theta is not None else None
            continue

        L1 = float(np.linalg.norm(elbow - shoulder))
        L2 = float(np.linalg.norm(wrist - elbow))
        if not (np.isfinite(L1) and np.isfinite(L2) and L1 > 1e-8 and L2 > 1e-8):
            continue

        if prev_theta_prior is not None and np.all(np.isfinite(prev_theta_prior[t])):
            theta_init = prev_theta_prior[t]
            prior_for_smoothing = prev_theta_prior[t]
        else:
            theta_init = (
                prev_theta if prev_theta is not None else np.zeros((4,), dtype=np.float64)
            )
            prior_for_smoothing = prev_theta

        theta = solve_ik_frame(
            shoulder,
            elbow,
            wrist,
            (L1, L2),
            theta_init=theta_init,
            prev_theta=prior_for_smoothing,
            smooth_w=smooth_w,
            limit_w=limit_w,
            elbow_w=elbow_w,
            wrist_w=wrist_w,
        )

        out[t] = theta
        prev_theta = theta

    return np.degrees(out)
