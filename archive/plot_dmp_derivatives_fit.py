"""
Plot DMP target vs fitted dq and ddq for one joint.

Examples:
  python3 vis/plot_dmp_derivatives_fit.py --subject 1 --motion lift --trial 6 --joint 1
  python3 vis/plot_dmp_derivatives_fit.py --path test_data/processed/subject_01/lift/trial_006 --joint 1 --derivative-method savgol
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on sys.path when launched directly.
_here = Path(__file__).resolve()
_project_root = _here.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dmp.dmp import (
    fit,
    rollout_simple,
    estimate_derivatives,
    canonical_phase,
)
from vis.plotting import load_angles_demo
from vis.trial_naming import trial_prefix


JOINT_NAMES = [
    "Elbow flexion",
    "Shoulder flexion",
    "Shoulder abduction",
    "Shoulder internal rotation",
]


def _load_meta(trial_dir: Path) -> dict:
    meta_path = trial_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def plot_dq_ddq_single_joint(
    q_demo: np.ndarray,
    joint_idx: int,
    out_path: Path,
    *,
    n_basis: int = 30,
    tau: float = 1.0,
    alpha_canonical: float = 4.0,
    alpha_transformation: float = 25.0,
    beta_transformation: float = 6.25,
    derivative_method: str = "savgol",
    savgol_window_length: int = 11,
    savgol_polyorder: int = 3,
    meta: dict | None = None,
    title_suffix: str = "",
) -> None:
    if q_demo.ndim != 2:
        raise ValueError(f"Expected q_demo shape (T, n_joints), got {q_demo.shape}")
    if not (0 <= joint_idx < q_demo.shape[1]):
        raise ValueError(f"joint_idx={joint_idx} out of bounds for q_demo with {q_demo.shape[1]} joints")

    T = int(q_demo.shape[0])
    dt_eff = tau / (T - 1)

    model = fit(
        [q_demo],
        tau=tau,
        dt=dt_eff,
        n_basis_functions=n_basis,
        alpha_canonical=alpha_canonical,
        alpha_transformation=alpha_transformation,
        beta_transformation=beta_transformation,
    )

    q_gen = rollout_simple(
        model,
        q0=q_demo[0],
        g=q_demo[-1],
        tau=tau,
        dt=dt_eff,
    )

    qj_tgt = q_demo[:, joint_idx]
    qj_gen = q_gen[:, joint_idx]

    dq_tgt, ddq_tgt = estimate_derivatives(
        qj_tgt,
        dt=dt_eff,
        derivative_method=derivative_method,
        savgol_window_length=savgol_window_length,
        savgol_polyorder=savgol_polyorder,
    )
    dq_gen, ddq_gen = estimate_derivatives(
        qj_gen,
        dt=dt_eff,
        derivative_method=derivative_method,
        savgol_window_length=savgol_window_length,
        savgol_polyorder=savgol_polyorder,
    )

    t = np.arange(T, dtype=np.float64) * dt_eff
    x = canonical_phase(t, tau=tau, alpha_canonical=alpha_canonical)

    joint_label = JOINT_NAMES[joint_idx] if joint_idx < len(JOINT_NAMES) else f"joint {joint_idx}"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    ax_dq_time, ax_dq_phase = axes[0, 0], axes[0, 1]
    ax_ddq_time, ax_ddq_phase = axes[1, 0], axes[1, 1]

    ax_dq_time.plot(t, dq_tgt, label="target dq", color="#2c3e50", linewidth=1.5)
    ax_dq_time.plot(t, dq_gen, label="DMP dq", color="#e67e22", linestyle="--", linewidth=1.5)
    ax_dq_time.set_title("dq vs time")
    ax_dq_time.set_ylabel("dq (rad/s)")
    ax_dq_time.grid(True, alpha=0.3)
    ax_dq_time.legend(loc="upper right")

    ax_ddq_time.plot(t, ddq_tgt, label="target ddq", color="#2c3e50", linewidth=1.5)
    ax_ddq_time.plot(t, ddq_gen, label="DMP ddq", color="#e67e22", linestyle="--", linewidth=1.5)
    ax_ddq_time.set_title("ddq vs time")
    ax_ddq_time.set_xlabel("time (s)")
    ax_ddq_time.set_ylabel("ddq (rad/s^2)")
    ax_ddq_time.grid(True, alpha=0.3)
    ax_ddq_time.legend(loc="upper right")

    ax_dq_phase.plot(x, dq_tgt, label="target dq", color="#2c3e50", linewidth=1.5)
    ax_dq_phase.plot(x, dq_gen, label="DMP dq", color="#e67e22", linestyle="--", linewidth=1.5)
    ax_dq_phase.set_title("dq vs phase x")
    ax_dq_phase.set_xlabel("phase x")
    ax_dq_phase.grid(True, alpha=0.3)
    ax_dq_phase.legend(loc="upper right")

    ax_ddq_phase.plot(x, ddq_tgt, label="target ddq", color="#2c3e50", linewidth=1.5)
    ax_ddq_phase.plot(x, ddq_gen, label="DMP ddq", color="#e67e22", linestyle="--", linewidth=1.5)
    ax_ddq_phase.set_title("ddq vs phase x")
    ax_ddq_phase.set_xlabel("phase x")
    ax_ddq_phase.set_ylabel("ddq (rad/s^2)")
    ax_ddq_phase.grid(True, alpha=0.3)
    ax_ddq_phase.legend(loc="upper right")

    m = meta or {}
    subject = m.get("subject", "?")
    motion = m.get("motion", "?")
    trial = m.get("trial", "?")

    suffix = f" ({title_suffix})" if title_suffix else ""

    fig.suptitle(
        f"DMP derivatives — subject {subject}, {motion}, trial {trial}, {joint_label}"
        f", n_basis={n_basis}{suffix}",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot DMP target vs fitted dq and ddq for a selected joint and trial."
    )
    parser.add_argument("--path", type=Path, default=None, help="Path to trial dir.")
    parser.add_argument("--subject", type=int, default=1, help="Subject number.")
    parser.add_argument("--motion", type=str, default="reach", help="Motion name.")
    parser.add_argument("--trial", type=int, default=1, help="Trial number.")
    parser.add_argument("--data-dir", type=Path, default=Path("test_data/processed"))
    parser.add_argument("--joint", type=int, default=1, help="Joint index (0..3).")
    parser.add_argument("--n-basis", type=int, default=30)
    parser.add_argument("--angles-source", type=str, default="auto", choices=["auto", "raw", "clean"])
    parser.add_argument("--derivative-method", type=str, default="savgol", choices=["gradient", "savgol"])
    parser.add_argument("--savgol-window-length", type=int, default=11)
    parser.add_argument("--savgol-polyorder", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)

    args = parser.parse_args()

    if args.path is not None:
        trial_dir = Path(args.path)
    else:
        trial_dir = args.data_dir / f"subject_{args.subject:02d}" / args.motion / f"trial_{args.trial:03d}"

    q_demo = load_angles_demo(trial_dir, source=args.angles_source)
    meta = _load_meta(trial_dir)

    if args.out is None:
        prefix = trial_prefix(trial_dir)
        out_path = trial_dir / f"{prefix}dmp_derivatives_fit_joint{args.joint}_n{args.n_basis}.png"
    else:
        out_path = Path(args.out)

    plot_dq_ddq_single_joint(
        q_demo=q_demo,
        joint_idx=int(args.joint),
        out_path=out_path,
        n_basis=int(args.n_basis),
        derivative_method=args.derivative_method,
        savgol_window_length=int(args.savgol_window_length),
        savgol_polyorder=int(args.savgol_polyorder),
        meta=meta,
        title_suffix=args.angles_source,
    )

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()