#!/usr/bin/env python3
"""
Record a movement trial, fit a DMP, then play it back in PyBullet (with table).

This is a convenience script that stitches together:
  1) capture (writes test_data/processed/.../left_arm_seq_camera.npy)
  2) keypoints -> angles (cleaned)
  3) retarget cleaned demo into robot joint ranges
  4) DMP fit + rollout (saved as dmp_rollout_clean.npz)
  5) simulation playback via sim/limb_sim_table.py

Notes
-----
- The simulation runs in "real-time" according to the rollout dt (time.sleep(dt) in sim).
- Requires the same dependencies as your capture + sim pipeline (e.g., depthai/mediapipe or marker pipeline, and pybullet).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dmp.dmp import fit, rollout_simple
from kinematics.clean_angles import _lowpass_angles
from kinematics.simple_kinematics import get_angles
from mapping.retarget import retarget


def _trial_dir(*, processed_dir: Path, subject: int, motion: str, trial: int) -> Path:
    return processed_dir / f"subject_{subject:02d}" / motion / f"trial_{trial:03d}"


def _load_raw_seq_t(trial_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    return np.load(trial_dir / "left_arm_seq_camera.npy"), np.load(trial_dir / "left_arm_t.npy")


def _load_meta(trial_dir: Path) -> dict:
    return json.load(open(trial_dir / "meta.json"))


def interpolate_nan(angles: np.ndarray) -> np.ndarray:
    """Interpolate NaNs over time, column-wise. Input: (T, D)."""
    a = np.asarray(angles, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"Expected angles shape (T, D), got {a.shape}")

    T, D = a.shape
    x = np.arange(T, dtype=np.float64)
    out = a.copy()
    for j in range(D):
        col = a[:, j]
        valid = np.isfinite(col)
        n_valid = int(np.sum(valid))
        if n_valid == 0:
            continue
        if n_valid == 1:
            out[:, j] = col[valid][0]
            continue
        out[:, j] = np.interp(x, x[valid], col[valid])
    return out


def _run_capture(*, subject: int, motion: str, trial: int, processed_dir: Path, fps: int) -> Path:
    """
    Run one of the existing capture scripts to produce a trial folder.
    Returns trial_dir.
    """
    trial_dir = _trial_dir(processed_dir=processed_dir, subject=subject, motion=motion, trial=trial)

    cmd = [
        sys.executable,
        "capture/3d_pose.py",
        "--subject",
        str(subject),
        "--motion",
        str(motion),
        "--trial",
        str(trial),
        "--processed-dir",
        str(processed_dir),
        "--fps",
        str(int(fps)),
    ]

    print("Recording capture trial...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    if not (trial_dir / "left_arm_seq_camera.npy").exists():
        raise FileNotFoundError(f"Capture did not produce left_arm_seq_camera.npy in {trial_dir}")
    return trial_dir


def _fit_rollout_and_save_clean_retarget(*, trial_dir: Path, fps: float, n_basis: int) -> None:
    """
    Build angles -> clean -> retarget demo -> fit+rollout -> save dmp_rollout_clean.npz.
    """
    seq, t = _load_raw_seq_t(trial_dir)
    _meta = _load_meta(trial_dir)

    angles_deg = get_angles(seq)
    angles_deg = interpolate_nan(angles_deg)
    clean_angles_deg = _lowpass_angles(angles_deg, fps=float(fps), cutoff_hz=5.0, order=2)

    # Retarget cleaned demo (degrees) into robot joint ranges.
    clean_angles_deg_retarget = retarget(clean_angles_deg)
    clean_demo = np.deg2rad(clean_angles_deg_retarget)  # (T,4) radians

    if clean_demo.shape[0] < 10:
        raise ValueError(f"Too few samples after cleaning: {clean_demo.shape}")

    tau = 1.0
    dt = tau / float(clean_demo.shape[0] - 1)

    model = fit(
        [clean_demo],
        tau=tau,
        dt=dt,
        n_basis_functions=int(n_basis),
        alpha_canonical=4.0,
        alpha_transformation=25.0,
        beta_transformation=6.25,
    )
    q_gen = rollout_simple(model, clean_demo[0], clean_demo[-1], tau=tau, dt=dt)

    out_path = trial_dir / "dmp_rollout_clean.npz"
    np.savez(
        out_path,
        q_demo=clean_demo,
        q_gen=np.rad2deg(q_gen),
        q_gen_rad=q_gen,
        t=t,
        dt=dt,
        q0=clean_demo[0],
        qT=clean_demo[-1],
    )
    print(f"Saved DMP rollout: {out_path}")


def _run_sim_table(*, trial_dir: Path) -> None:
    cmd = [sys.executable, "sim/limb_sim_table.py", "--path", str(trial_dir), "--source", "clean"]
    print("Starting simulation...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Record movement, fit DMP, and simulate in PyBullet (with table).")
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--motion", type=str, required=True)
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "test_data" / "processed")
    ap.add_argument("--fps", type=int, default=25, help="FPS used for capture + filtering assumptions.")
    args = ap.parse_args()

    # 1) Record capture
    trial_dir = _run_capture(
        subject=int(args.subject),
        motion=str(args.motion),
        trial=int(args.trial),
        processed_dir=Path(args.processed_dir),
        fps=int(args.fps),
    )

    # 2) Fit DMP (cleaned demo retargeted) + save rollout
    _fit_rollout_and_save_clean_retarget(trial_dir=trial_dir, fps=float(args.fps), n_basis=30)

    # 3) Simulate (with table)
    _run_sim_table(trial_dir=trial_dir)


if __name__ == "__main__":
    main()

