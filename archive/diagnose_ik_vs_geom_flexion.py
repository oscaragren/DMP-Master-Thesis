"""
Quick diagnostic: compare IK shoulder flexion vs geometric flexion.

For each timestep:
  - u = elbow - shoulder
  - Build a trunk frame:
      * If MediaPipe-full landmarks are available (N>=25), use shoulders + hips.
      * Otherwise, use identity (camera frame) as requested.
  - u_local = R_trunk.T @ u
  - flex_geom = deg(arctan2(u_local_z, -u_local_y))
  - IK flexion = angles_ik[:, 0] from kinematics.left_arm_angles.shoulder_angles(..., method="ik")

Usage examples:
  python3 tests/diagnose_ik_vs_geom_flexion.py --trial test_data/processed/subject_01/reach/trial_014
  python3 tests/diagnose_ik_vs_geom_flexion.py --seq test_data/processed/subject_01/reach/trial_014/left_arm_seq_camera.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on sys.path so `kinematics.*` imports work when run from tests/
_here = Path(__file__).resolve()
_project_root = _here.parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from kinematics.left_arm_angles import shoulder_angles


MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_ELBOW = 13
MP_LEFT_WRIST = 15


def _normalize_rows(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    out = np.full_like(v, np.nan, dtype=np.float64)
    ok = n.squeeze(1) > eps
    out[ok] = v[ok] / n[ok]
    return out


def _build_trunk_frame_shoulders_hips(
    ls: np.ndarray, rs: np.ndarray, lh: np.ndarray, rh: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build per-frame trunk rotation matrices R_trunk (T,3,3) in camera coords.

    Axes:
      x: person right  (rs - ls)
      y: person up     (shoulder_center - hip_center)
      z: person forward = x × y

    Returns:
      R: (T,3,3) columns are [x, y, z]
      valid: (T,) boolean
    """
    ls = np.asarray(ls, dtype=np.float64)
    rs = np.asarray(rs, dtype=np.float64)
    lh = np.asarray(lh, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)

    x = rs - ls
    x_u = _normalize_rows(x)

    sh_c = 0.5 * (ls + rs)
    hip_c = 0.5 * (lh + rh)
    y = sh_c - hip_c
    # Make y orthogonal to x
    y = y - np.einsum("ij,ij->i", y, x_u)[:, None] * x_u
    y_u = _normalize_rows(y)

    # Match project convention: +Z is forward out of chest.
    z = -np.cross(x_u, y_u)
    if np.dot(z, np.array([0, 0, 1])) < 0:
        z = -z
    z_u = _normalize_rows(z)

    # Re-orthogonalize y to ensure orthonormal basis
    y_u = _normalize_rows(np.cross(z_u, x_u))

    R = np.stack([x_u, y_u, z_u], axis=2)
    valid = np.all(np.isfinite(R.reshape(R.shape[0], -1)), axis=1)
    return R, valid


def _infer_and_extract_left_arm(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (shoulder, elbow, wrist, right_shoulder) as (T,3) each.

    Supports:
      - Compact left-arm sequences: (T,4,3) with [LS, LE, LW, RS]
      - MediaPipe-full sequences: (T,33,3) (or larger), using landmark indices
    """
    if seq.ndim != 3 or seq.shape[2] != 3:
        raise ValueError(f"Expected seq shape (T,N,3), got {seq.shape}")

    if seq.shape[1] >= 25:
        S = seq[:, MP_LEFT_SHOULDER, :]
        E = seq[:, MP_LEFT_ELBOW, :]
        W = seq[:, MP_LEFT_WRIST, :]
        RS = seq[:, MP_RIGHT_SHOULDER, :]
        return S, E, W, RS

    if seq.shape[1] >= 4:
        return seq[:, 0, :], seq[:, 1, :], seq[:, 2, :], seq[:, 3, :]

    raise ValueError(f"Need at least 4 keypoints per frame, got N={seq.shape[1]}")


def _compute_local_vectors_and_flex_geom(
    seq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute:
      - u_local: (T,3) upper-arm vector in trunk/local frame
      - w_local: (T,3) wrist-from-shoulder vector in trunk/local frame
      - flex_geom: (T,) geometric flexion (deg) from u_local
      - u_local_z: (T,) forward component of u_local
    """
    S, E, W, _ = _infer_and_extract_left_arm(seq)
    u = E - S  # shoulder -> elbow
    w = W - S  # shoulder -> wrist

    # Build trunk frame using shoulders+hips if available, else identity.
    if seq.shape[1] >= 25:
        ls = seq[:, MP_LEFT_SHOULDER, :]
        rs = seq[:, MP_RIGHT_SHOULDER, :]
        lh = seq[:, MP_LEFT_HIP, :]
        rh = seq[:, MP_RIGHT_HIP, :]
        R_trunk, valid_R = _build_trunk_frame_shoulders_hips(ls, rs, lh, rh)
    else:
        T = seq.shape[0]
        R_trunk = np.broadcast_to(np.eye(3, dtype=np.float64), (T, 3, 3)).copy()
        valid_R = np.ones((T,), dtype=bool)

    u_local = np.full_like(u, np.nan, dtype=np.float64)
    w_local = np.full_like(w, np.nan, dtype=np.float64)
    valid_u = np.all(np.isfinite(u), axis=1) & valid_R
    if np.any(valid_u):
        u_local[valid_u] = np.einsum("tji,tj->ti", R_trunk[valid_u], u[valid_u])
    valid_w = np.all(np.isfinite(w), axis=1) & valid_R
    if np.any(valid_w):
        w_local[valid_w] = np.einsum("tji,tj->ti", R_trunk[valid_w], w[valid_w])

    flex_geom = np.full((seq.shape[0],), np.nan, dtype=np.float64)
    ok = np.all(np.isfinite(u_local), axis=1)
    if np.any(ok):
        flex_geom[ok] = np.degrees(np.arctan2(u_local[ok, 2], -u_local[ok, 1]))

    return u_local, w_local, flex_geom, u_local[:, 2]


def _load_seq_and_time(trial_or_seq: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a sequence and timestamps.

    If given a trial directory, loads:
      - left_arm_seq_camera_cleaned.npy if present, else left_arm_seq_camera.npy
      - timestamps from left_arm_t_cleaned.npy / left_arm_t.npy if present, else index.
    If given a .npy, loads that file and tries to find a sibling t file.
    """
    p = Path(trial_or_seq)
    if p.is_dir():
        seq_path = p / "left_arm_seq_camera_cleaned.npy"
        t_path = p / "left_arm_t_cleaned.npy"
        if not seq_path.exists():
            seq_path = p / "left_arm_seq_camera.npy"
            t_path = p / "left_arm_t.npy"
        seq = np.load(seq_path)
        if t_path.exists():
            t = np.load(t_path).astype(np.float64, copy=False)
        else:
            t = np.arange(seq.shape[0], dtype=np.float64)
        if t.size > 0:
            t = t - t[0]
        return seq, t

    if p.suffix.lower() == ".npy":
        seq = np.load(p)
        # sibling timestamps (best-effort)
        t = None
        if p.name.endswith("_seq_camera.npy"):
            cand = p.with_name(p.name.replace("_seq_camera.npy", "_t.npy"))
            if cand.exists():
                t = np.load(cand).astype(np.float64, copy=False)
        if t is None:
            t = np.arange(seq.shape[0], dtype=np.float64)
        if t.size > 0:
            t = t - t[0]
        return seq, t

    raise ValueError(f"Provide a trial directory or a .npy path, got: {p}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic plot: geometric shoulder flexion vs IK flexion."
    )
    parser.add_argument(
        "--trial",
        type=Path,
        default=None,
        help="Trial directory containing left_arm_seq_camera*.npy",
    )
    parser.add_argument(
        "--seq",
        type=Path,
        default=None,
        help="Direct path to a sequence .npy (T,N,3)",
    )
    parser.add_argument(
        "--ik-use-trunk-frame",
        action="store_true",
        help="Run IK after expressing [shoulder, elbow, wrist] in the shoulder-based trunk frame.",
    )
    parser.add_argument(
        "--plot-u-local-z",
        action="store_true",
        help="Also plot u_local[:,2] (forward component in trunk/local frame).",
    )
    parser.add_argument(
        "--plot-local-components",
        action="store_true",
        help="Also save/show a second plot with u_local[:,0/1/2] and w_local[:,2].",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Save plot to this path (PNG). If omitted, shows interactively.",
    )
    args = parser.parse_args()

    if (args.trial is None) == (args.seq is None):
        raise SystemExit("Provide exactly one of --trial or --seq")

    src = args.trial if args.trial is not None else args.seq
    seq, t = _load_seq_and_time(Path(src))

    # Geometric flexion based on requested formula
    u_local, w_local, flex_geom, u_local_z = _compute_local_vectors_and_flex_geom(seq)

    # IK flexion from existing solver API
    # shoulder_angles(..., method="ik") expects (T,N,3) with at least [0]=S,[1]=E,[2]=W
    # plus [3]=right_shoulder if ik_use_trunk_frame=True.
    if seq.shape[1] >= 4:
        seq_for_ik = seq[:, :4, :] if seq.shape[1] < 25 else seq  # keep full if present
    else:
        seq_for_ik = seq

    angles_ik = shoulder_angles(
        seq_for_ik,
        method="ik",
        ik_use_trunk_frame=bool(args.ik_use_trunk_frame),
    )  # (T,4) deg
    flex_ik = angles_ik[:, 0]

    fig, ax = plt.subplots(1, 1, figsize=(11, 4.5), sharex=True)
    ok = np.isfinite(flex_geom)
    if np.any(ok):
        ax.plot(t[ok], flex_geom[ok], label="flex_geom (deg)", color="#e74c3c", linewidth=1.5)
    ok = np.isfinite(flex_ik)
    if np.any(ok):
        ax.plot(t[ok], flex_ik[ok], label="IK flexion angles[:,0] (deg)", color="#3498db", linewidth=1.5)

    if args.plot_u_local_z:
        ok = np.isfinite(u_local_z)
        if np.any(ok):
            ax2 = ax.twinx()
            ax2.plot(t[ok], u_local_z[ok], label="u_local[:,2] (forward)", color="#2ecc71", alpha=0.7, linewidth=1.2)
            ax2.set_ylabel("u_local z (m)")
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        else:
            ax.legend(loc="upper right")
    else:
        ax.legend(loc="upper right")

    ax.set_title("Shoulder flexion: geometric vs IK")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)

    plt.tight_layout()
    saved_main = False
    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"Saved {out_path}")
        saved_main = True

    # Optional second plot: local components (best next test)
    if args.plot_local_components:
        fig2, axc = plt.subplots(1, 1, figsize=(11, 4.5), sharex=True)
        labels = ["u_local x (right)", "u_local y (up)", "u_local z (forward)", "w_local z (wrist forward)"]
        series = [u_local[:, 0], u_local[:, 1], u_local[:, 2], w_local[:, 2]]
        colors = ["#f39c12", "#8e44ad", "#2ecc71", "#16a085"]
        for vals, lab, col in zip(series, labels, colors):
            ok = np.isfinite(vals)
            if np.any(ok):
                axc.plot(t[ok], vals[ok], label=lab, color=col, linewidth=1.3)
        axc.set_title("Local components: upper-arm and wrist forward movement")
        axc.set_xlabel("Time (s)")
        axc.set_ylabel("Local component (m)")
        axc.grid(True, alpha=0.3)
        axc.legend(loc="upper right")
        plt.tight_layout()

        if args.out is not None:
            out2 = out_path.with_name(out_path.stem + "_components" + out_path.suffix)
            plt.savefig(out2, dpi=160)
            plt.close(fig2)
            print(f"Saved {out2}")
        else:
            plt.show()

    if not saved_main and args.out is None:
        plt.show()


if __name__ == "__main__":
    main()

