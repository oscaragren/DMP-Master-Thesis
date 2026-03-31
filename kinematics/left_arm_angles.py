import numpy as np
from scipy.spatial.transform import Rotation as R

# Trunk "up" in camera frame (OAK-D: X right, Y down, Z forward → person up = -Y)
WORLD_UP = np.array([0.0, -1.0, 0.0], dtype=np.float64)


def limb_vectors(seq: np.ndarray) -> np.ndarray:
    """
    seq: (T, 4, 3) with [shoulder, elbow, wrist, right_shoulder] in meters
    returns: 
        u: (T, 3) with elbow -> shoulder vector
        v: (T, 3) with elbow -> wrist vector
    """
    S = seq[:, 0, :]
    E = seq[:, 1, :]
    W = seq[:, 2, :]
    return S - E, E - W

def _elbow_flexion_rad_core(seq: np.ndarray) -> np.ndarray:
    """
    Core computation of elbow flexion in radians.

    seq: (T, 4, 3) with [shoulder, elbow, wrist, right_shoulder] in meters
    returns: (T,) elbow angle in radians, NaN where missing.
    """
    u, v = limb_vectors(seq)

    # Normalize with NaN safety
    u_norm = np.linalg.norm(u, axis=1)
    v_norm = np.linalg.norm(v, axis=1)
    valid = (
        (u_norm > 0)
        & (v_norm > 0)
        & np.all(np.isfinite(u), axis=1)
        & np.all(np.isfinite(v), axis=1)
    )

    out = np.full((seq.shape[0],), np.nan, dtype=np.float64)
    dot = np.einsum("ij,ij->i", u, v)
    cosang = np.clip(dot / (u_norm * v_norm), -1.0, 1.0)
    out[valid] = np.arccos(cosang[valid])
    return out


def elbow_flexion_rad(seq: np.ndarray) -> np.ndarray:
    """
    Elbow flexion in radians.

    seq: (T, 4, 3) with [shoulder, elbow, wrist, right_shoulder] in meters
    returns: (T,) elbow angle in radians, NaN where missing.
    """
    return _elbow_flexion_rad_core(seq)


def elbow_flexion_deg(seq: np.ndarray) -> np.ndarray:
    """
    Elbow flexion in degrees (legacy helper, derived from radians).

    seq: (T, 4, 3) with [shoulder, elbow, wrist, right_shoulder] in meters
    returns: (T,) elbow angle in degrees, NaN where missing.
    """
    return np.degrees(_elbow_flexion_rad_core(seq))


def _trunk_frame_from_shoulders(
    left_shoulder: np.ndarray, right_shoulder: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build right-handed trunk frame at left shoulder.
    left_shoulder, right_shoulder: (T, 3)
    Returns x_axis (T,3), y_axis (T,3), z_axis (T,3) unit vectors.
    x = right_shoulder - left_shoulder (normalized), y ≈ up, z = forward.
    """
    x = right_shoulder - left_shoulder  # vector between shoulders
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    valid = (x_norm.squeeze(1) > 1e-8) & np.all(np.isfinite(x), axis=1)
    x = np.where(x_norm > 1e-8, x / x_norm, np.nan)

    # y = up (world) projected onto plane perpendicular to x
    up = np.broadcast_to(WORLD_UP, (x.shape[0], 3))
    y = up - np.einsum("ij,ij->i", up, x)[:, None] * x  # project up vector onto plane perpendicular to x
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    y = np.where(y_norm > 1e-8, y / y_norm, np.nan)
    
    # z = forward (world) projected onto plane perpendicular to x and y
    z = np.cross(x, y)  # cross product of x and y to get z axis
    z_norm = np.linalg.norm(z, axis=1, keepdims=True)
    z = np.where(z_norm > 1e-8, z / z_norm, np.nan)
    return x, y, z  # x, y, z axes of trunk frame


def _normalize_rows(v: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """
    Row-wise normalization with NaN safety.

    Args:
        v: (T, 3)
        eps: threshold below which a row is considered degenerate

    Returns:
        v_u: (T, 3) unit vectors (NaN where degenerate)
        n: (T,) norms
    """
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=1)
    v_u = np.where((n > eps)[:, None], v / np.maximum(n[:, None], eps), np.nan)
    return v_u, n


def trunk_rotation_matrix(seq: np.ndarray) -> np.ndarray:
    """
    Build trunk rotation matrix R_t from shoulder landmarks.

    seq: (T, N, 3) with at least:
        0: left_shoulder
        3: right_shoulder

    Returns:
        R_t: (T, 3, 3) rotation matrix with columns [x_t, y_t, z_t].
        NaN for frames where trunk axes are invalid/degenerate.
    """
    if seq.ndim != 3 or seq.shape[2] != 3 or seq.shape[1] < 4:
        raise ValueError(f"Expected seq shape (T, N>=4, 3), got {seq.shape}")
    ls = seq[:, 0, :]
    rs = seq[:, 3, :]
    x_t, y_t, z_t = _trunk_frame_from_shoulders(ls, rs)
    R_t = np.stack([x_t, y_t, z_t], axis=2)  # (T,3,3) columns are axes
    return R_t


def _upper_arm_rotation_matrix_from_hand(seq: np.ndarray, R_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an upper-arm frame rotation matrix R_a using elbow direction + hand direction.

    Required indices for rotmat method:
        0: left_shoulder
        1: left_elbow
        2: left_wrist
        4: left_index
        5: left_pinky

    Frame definition (columns):
        y_a: shoulder -> elbow (long axis)
        x_a: hand direction projected orthogonal to y_a (twist-resolving axis)
        z_a: x_a × y_a

    Falls back to a trunk-based direction when hand projection degenerates.

    Returns:
        R_a: (T, 3, 3) with columns [x_a, y_a, z_a]
        valid: (T,) boolean validity mask for frames
    """
    T = seq.shape[0]

    S = seq[:, 0, :]
    E = seq[:, 1, :]
    W = seq[:, 2, :]
    I = seq[:, 4, :]
    P = seq[:, 5, :]

    # y_a = shoulder -> elbow
    y_a, y_n = _normalize_rows(E - S)

    # hand_dir = normalize((index+pinky)/2 - wrist)
    hand_mid = 0.5 * (I + P)
    hand_dir, hand_n = _normalize_rows(hand_mid - W)

    # x_a_raw = hand_dir projected onto plane orthogonal to y_a
    dot_hy = np.einsum("ij,ij->i", hand_dir, y_a)
    x_a_raw = hand_dir - dot_hy[:, None] * y_a
    x_a, x_n = _normalize_rows(x_a_raw)

    # Fallback: if x_a degenerate, use trunk z-axis projected orthogonal to y_a
    z_t = R_t[:, :, 2]
    dot_zy = np.einsum("ij,ij->i", z_t, y_a)
    x_fallback_raw = z_t - dot_zy[:, None] * y_a
    x_fallback, x_fb_n = _normalize_rows(x_fallback_raw)
    use_fb = ~(x_n > 1e-8)
    x_a = np.where(use_fb[:, None], x_fallback, x_a)

    # z_a = x_a × y_a, then re-orthogonalize x_a = y_a × z_a
    z_a, z_n = _normalize_rows(np.cross(x_a, y_a))
    x_a, _ = _normalize_rows(np.cross(y_a, z_a))

    R_a = np.stack([x_a, y_a, z_a], axis=2)

    valid = (
        np.all(np.isfinite(S), axis=1)
        & np.all(np.isfinite(E), axis=1)
        & np.all(np.isfinite(W), axis=1)
        & np.all(np.isfinite(I), axis=1)
        & np.all(np.isfinite(P), axis=1)
        & np.all(np.isfinite(R_t.reshape(T, -1)), axis=1)
        & (y_n > 1e-8)
        & (z_n > 1e-8)
        & ((x_n > 1e-8) | (x_fb_n > 1e-8))
        & (hand_n > 1e-8)
    )

    return R_a, valid


def shoulder_angles_rotmat(seq: np.ndarray) -> np.ndarray:
    """
    Rotation-matrix shoulder angles using hand direction to resolve twist.

    This is intended to provide a more stable *axial rotation proxy* than the
    vector-only method, which is underdetermined from shoulder–elbow–wrist alone.

    Input:
        seq: (T, N, 3) with indices:
            0: left_shoulder
            1: left_elbow
            2: left_wrist
            3: right_shoulder
            4: left_index
            5: left_pinky

    Steps:
        1) Build trunk frame rotation matrix R_t (columns are trunk axes).
        2) Build upper-arm frame R_a using y_a = shoulder->elbow and x_a from hand direction.
        3) Relative rotation: R_rel = R_t^T @ R_a
        4) Euler extraction: rot.as_euler("ZXY", degrees=True)

    Output:
        (T, 3) degrees: [flexion_deg, abduction_deg, axial_rotation_proxy_deg]

    Note on mapping:
        SciPy returns angles in the same order as the sequence string. For "ZXY",
        `as_euler` returns [ang_Z, ang_X, ang_Y]. We map:
            flexion  = ang_X
            abduction = ang_Z
            axial_rotation_proxy = ang_Y
        The exact anatomical correspondence depends on the trunk/arm frame conventions.
    """
    if seq.ndim != 3 or seq.shape[2] != 3:
        raise ValueError(f"Expected seq shape (T, N, 3), got {seq.shape}")
    if seq.shape[1] < 6:
        out = np.full((seq.shape[0], 3), np.nan, dtype=np.float64)
        return out

    T = seq.shape[0]
    R_t = trunk_rotation_matrix(seq)
    R_a, valid = _upper_arm_rotation_matrix_from_hand(seq, R_t=R_t)

    R_rel = np.einsum("tij,tjk->tik", np.transpose(R_t, (0, 2, 1)), R_a)

    out = np.full((T, 3), np.nan, dtype=np.float64)
    if np.any(valid):
        rot = R.from_matrix(R_rel[valid])
        eul = rot.as_euler("ZXY", degrees=True)  # (n_valid, 3) => [Z, X, Y]
        out[valid, 0] = eul[:, 1]  # flexion
        out[valid, 1] = eul[:, 0]  # abduction
        out[valid, 2] = eul[:, 2]  # axial rotation proxy
    return out


def shoulder_angles(
    seq: np.ndarray,
    method: str = "vector",
    prev_angles: np.ndarray | None = None,
) -> np.ndarray:
    """
    Unified left-arm angle API with selectable method.

    Args:
        seq: (T, N, 3) keypoints in camera frame.
            Minimum required indices (both methods):
                0: left_shoulder
                1: left_elbow
                2: left_wrist
                3: right_shoulder
            Additional required for method="rotmat":
                4: left_index
                5: left_pinky
        method:
            - "vector": existing vector-geometry method (fast, simple).
              The returned internal rotation is **not reliable** as a true anatomical internal rotation
              because twist is underdetermined from shoulder–elbow–wrist alone.
            - "rotmat": rotation-matrix method using hand direction to resolve twist.
              Returns an **axial_rotation_proxy** (still a proxy; not pure anatomical internal rotation).
            - "ik": optimization-based IK with temporal smoothing + soft joint limits.
              This does NOT directly measure internal rotation; it returns a **physically plausible
              estimate** stabilized by smoothness, joint limits, and kinematic consistency.
        prev_angles:
            Optional (T, 4) degrees prior used for method="ik" smoothing/initialization:
            [flex, abd, internal_rot, elbow_flex]. Ignored for other methods.

    Returns:
        (T, 4) degrees:
            [shoulder_flexion_deg, shoulder_abduction_deg, shoulder_internal_rotation_deg, elbow_flexion_deg]

    Fallback behavior:
        If method="rotmat" but hand landmarks are missing (N < 6), this function
        falls back to method="vector".
    """
    m = (method or "vector").strip().lower()
    if m not in {"vector", "rotmat", "ik"}:
        raise ValueError(f"Unknown method '{method}'. Use 'vector', 'rotmat', or 'ik'.")

    # Graceful fallback if hand landmarks are missing.
    if m == "rotmat" and (seq.ndim != 3 or seq.shape[1] < 6):
        m = "vector"

    if m == "vector":
        if seq.ndim != 3 or seq.shape[1] < 4 or seq.shape[2] != 3:
            raise ValueError(f"Expected seq shape (T, N>=4, 3) for vector method, got {seq.shape}")
        sh = shoulder_flex_abd_rot_3dof(seq[:, :4, :])  # (T,3)
        el = elbow_flexion_deg(seq[:, :4, :])  # (T,)
        return np.column_stack([sh, el])

    if m == "rotmat":
        sh = shoulder_angles_rotmat(seq)  # (T,3)
        el = elbow_flexion_deg(seq[:, :4, :])  # (T,)
        return np.column_stack([sh, el])

    # IK method: uses only shoulder, elbow, wrist (indices 0..2).
    from kinematics.ik_solver import solve_ik_sequence

    if seq.ndim != 3 or seq.shape[2] != 3 or seq.shape[1] < 3:
        raise ValueError(f"Expected seq shape (T, N>=3, 3) for ik method, got {seq.shape}")
    return solve_ik_sequence(seq[:, :3, :], prev_angles_deg=prev_angles)


def _signed_angle_around_axis(
    v_ref: np.ndarray, v_arm: np.ndarray, axis: np.ndarray
) -> np.ndarray:
    """
    Signed angle in radians from v_ref to v_arm around axis (all (T,3)).
    v_ref: (T, 3) reference vector
    v_arm: (T, 3) arm vector
    axis: (T, 3) axis vector
    returns: (T,) signed angle in radians
    """
    # Project into plane perpendicular to axis
    axis_norm = np.linalg.norm(axis, axis=1, keepdims=True)
    axis_u = np.where(axis_norm > 1e-8, axis / axis_norm, np.nan)

    v_ref_p = v_ref - np.einsum("ij,ij->i", v_ref, axis_u)[:, None] * axis_u  # project v_ref onto plane perpendicular to axis
    v_arm_p = v_arm - np.einsum("ij,ij->i", v_arm, axis_u)[:, None] * axis_u  # project v_arm onto plane perpendicular to axis

    n_ref = np.linalg.norm(v_ref_p, axis=1)
    n_arm = np.linalg.norm(v_arm_p, axis=1)
    valid = (n_ref > 1e-8) & (n_arm > 1e-8)
    
    v_ref_p = np.where(valid[:, None], v_ref_p / np.maximum(n_ref[:, None], 1e-8), np.nan)
    v_arm_p = np.where(valid[:, None], v_arm_p / np.maximum(n_arm[:, None], 1e-8), np.nan)

    dot = np.einsum("ij,ij->i", v_ref_p, v_arm_p)  # dot product of v_ref_p and v_arm_p to get the cosine of the angle
    cross = np.cross(v_ref_p, v_arm_p)  # cross product of v_ref_p and v_arm_p to get the direction of the angle

    sign = np.sign(np.einsum("ij,ij->i", cross, axis_u))
    sign = np.where(sign == 0, 1, sign)  # if sign is 0, set to 1
    out = np.full((axis.shape[0],), np.nan, dtype=np.float64)
    out[valid] = sign[valid] * np.arccos(np.clip(dot[valid], -1.0, 1.0))  # angle in radians
    return out  # signed angle in radians


def shoulder_angles_3dof(seq: np.ndarray) -> np.ndarray:
    """
    Compute 3-DOF shoulder angles (elevation, plane of elevation, internal rotation).

    seq: (T, 4, 3) with [left_shoulder, left_elbow, left_wrist, right_shoulder] in meters (camera frame).
    returns: (T, 3) in degrees — [elevation_deg, azimuth_deg, internal_rotation_deg].
    NaN where keypoints are missing or degenerate.

    Convention:
    - Elevation: angle of upper arm from vertical (trunk up). 0° = arm up, 90° = horizontal, 180° = arm down.
    - Azimuth (plane of elevation): angle in the horizontal plane. 0° = forward (trunk z), positive = toward right.
    - Internal rotation: rotation of the humerus about its long axis (elbow flex axis vs reference plane).
    """
    T = seq.shape[0]
    ls = seq[:, 0, :]  # left shoulder
    rs = seq[:, 3, :]  # right shoulder

    upper_arm, forearm = limb_vectors(seq)

    x_trunk, y_trunk, z_trunk = _trunk_frame_from_shoulders(ls, rs)

    valid = (
        np.all(np.isfinite(seq), axis=(1, 2))
        & (np.linalg.norm(upper_arm, axis=1) > 1e-8)
        & (np.linalg.norm(forearm, axis=1) > 1e-8)
        & np.all(np.isfinite(x_trunk), axis=1)
        & np.all(np.isfinite(y_trunk), axis=1)
        & np.all(np.isfinite(z_trunk), axis=1)
    )

    out = np.full((T, 3), np.nan, dtype=np.float64)

    # Elevation: angle between upper arm and vertical (y_trunk)
    cos_el = np.einsum("ij,ij->i", upper_arm, y_trunk) / np.maximum(
        np.linalg.norm(upper_arm, axis=1), 1e-8
    )  # cosine of the angle between upper arm and vertical
    elevation_rad = np.arccos(np.clip(cos_el, -1.0, 1.0))  # angle in radians
    out[:, 0] = np.degrees(elevation_rad)  # angle in degrees

    # Azimuth: angle of upper-arm projection in horizontal (xz) plane
    ua_xz = upper_arm - np.einsum("ij,ij->i", upper_arm, y_trunk)[:, None] * y_trunk
    proj_x = np.einsum("ij,ij->i", ua_xz, x_trunk)  # projection of upper arm onto x axis
    proj_z = np.einsum("ij,ij->i", ua_xz, z_trunk)  # projection of upper arm onto z axis
    azimuth_rad = np.arctan2(proj_z, proj_x)
    out[:, 1] = np.degrees(azimuth_rad)

    # Internal rotation: signed angle from reference plane (upper_arm, y_trunk) to arm plane (upper_arm, forearm) about upper_arm
    n_ref = np.cross(upper_arm, y_trunk)
    n_arm = np.cross(upper_arm, forearm)
    internal_rad = _signed_angle_around_axis(n_ref, n_arm, upper_arm)
    out[:, 2] = np.degrees(internal_rad)

    out[~valid, :] = np.nan
    return out


def shoulder_flex_abd_rot_3dof(seq: np.ndarray) -> np.ndarray:
    """
    Compute 3-DOF shoulder angles (flexion/extension, abduction/adduction, internal rotation).

    seq: (T, 4, 3) with [left_shoulder, left_elbow, left_wrist, right_shoulder] in meters (camera frame).

    Returns:
        (T, 3) in degrees:
            [flexion_deg, abduction_deg, internal_rotation_deg]
        NaN where keypoints are missing or degenerate.

    Conventions (using trunk frame x,y,z from shoulders):
        - Flexion/extension:
            * Measured in the sagittal plane spanned by (y_trunk, z_trunk).
            * 0° ≈ arm aligned with +y_trunk (up), +90° ≈ arm forward (+z_trunk).
        - Abduction/adduction:
            * Measured in the frontal plane spanned by (y_trunk, x_trunk).
            * 0° ≈ arm aligned with +y_trunk (up), +90° ≈ arm out to the side (+x_trunk).
        - Internal rotation:
            * Same definition as in shoulder_angles_3dof — twist of the humerus about its long axis.
    """
    T = seq.shape[0]
    ls = seq[:, 0, :]  # left shoulder
    rs = seq[:, 3, :]  # right shoulder

    upper_arm, forearm = limb_vectors(seq)

    x_trunk, y_trunk, z_trunk = _trunk_frame_from_shoulders(ls, rs)

    valid = (
        np.all(np.isfinite(seq), axis=(1, 2))
        & (np.linalg.norm(upper_arm, axis=1) > 1e-8)
        & (np.linalg.norm(forearm, axis=1) > 1e-8)
        & np.all(np.isfinite(x_trunk), axis=1)
        & np.all(np.isfinite(y_trunk), axis=1)
        & np.all(np.isfinite(z_trunk), axis=1)
    )

    out = np.full((T, 3), np.nan, dtype=np.float64)

    # Flexion/extension: angle of upper-arm projection in sagittal (y,z) plane.
    # Remove left-right component (x_trunk) to stay in sagittal plane.
    ua_sag = upper_arm - np.einsum("ij,ij->i", upper_arm, x_trunk)[:, None] * x_trunk
    comp_y = np.einsum("ij,ij->i", ua_sag, y_trunk)
    comp_z = np.einsum("ij,ij->i", ua_sag, z_trunk)
    flex_rad = np.arctan2(comp_z, comp_y)
    out[:, 0] = np.degrees(flex_rad)

    # Abduction/adduction: angle of upper-arm projection in frontal (y,x) plane.
    # Remove forward/back component (z_trunk) to stay in frontal plane.
    ua_front = upper_arm - np.einsum("ij,ij->i", upper_arm, z_trunk)[:, None] * z_trunk
    comp_y_f = np.einsum("ij,ij->i", ua_front, y_trunk)
    comp_x = np.einsum("ij,ij->i", ua_front, x_trunk)
    abd_rad = np.arctan2(comp_x, comp_y_f)
    out[:, 1] = np.degrees(abd_rad)

    # Internal rotation: identical to shoulder_angles_3dof.
    n_ref = np.cross(upper_arm, y_trunk)
    n_arm = np.cross(upper_arm, forearm)
    internal_rad = _signed_angle_around_axis(n_ref, n_arm, upper_arm)
    out[:, 2] = np.degrees(internal_rad)

    out[~valid, :] = np.nan
    return out


def shoulder_flex_abd_rot_3dof_rad(seq: np.ndarray) -> np.ndarray:
    """
    Radian version of shoulder_flex_abd_rot_3dof.

    Returns:
        (T, 3) in radians:
            [flexion_rad, abduction_rad, internal_rotation_rad]
    """
    return np.deg2rad(shoulder_flex_abd_rot_3dof(seq))
