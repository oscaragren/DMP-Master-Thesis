"""
Plot DMP target forcing term vs fitted forcing term for one joint.

Examples (from project root):
  python3 vis/plot_dmp_forcing_fit.py --subject 1 --motion lift --trial 6 --joint 1
  python3 vis/plot_dmp_forcing_fit.py --path test_data/processed/subject_01/lift/trial_006 --joint 0 --n-basis 60
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


# Ensure project root is on sys.path when launched directly.
_here = Path(__file__).resolve()
_project_root = _here.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from vis.plotting import plot_dmp_forcing_fit_from_trial


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot DMP target forcing vs fitted forcing for a selected joint and trial.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to trial dir (overrides subject/motion/trial).",
    )
    parser.add_argument("--subject", type=int, default=1, help="Subject number.")
    parser.add_argument("--motion", type=str, default="reach", help="Motion name.")
    parser.add_argument("--trial", type=int, default=1, help="Trial number.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("test_data/processed"),
        help="Root directory (subject/motion/trial underneath).",
    )
    parser.add_argument(
        "--joint",
        type=int,
        default=1,
        help="Joint index: 0=elbow, 1=shoulder_flexion, 2=shoulder_abduction, 3=shoulder_internal_rotation.",
    )
    parser.add_argument("--n-basis", type=int, default=30, help="Number of DMP basis functions.")
    parser.add_argument(
        "--angles-source",
        type=str,
        default="auto",
        choices=["auto", "raw", "clean"],
        help="Which angle source to load from trial.",
    )
    parser.add_argument(
        "--derivative-method",
        type=str,
        default="gradient",
        choices=["gradient", "savgol"],
        help="Method for derivative estimation used in target forcing computation.",
    )
    parser.add_argument(
        "--savgol-window-length",
        type=int,
        default=11,
        help="Savitzky-Golay window length (odd). Used only with --derivative-method savgol.",
    )
    parser.add_argument(
        "--savgol-polyorder",
        type=int,
        default=3,
        help="Savitzky-Golay polynomial order. Used only with --derivative-method savgol.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output PNG path. Defaults to trial directory with trial prefix.",
    )
    args = parser.parse_args()

    if args.path is not None:
        trial_dir = Path(args.path)
    else:
        trial_dir = args.data_dir / f"subject_{args.subject:02d}" / args.motion / f"trial_{args.trial:03d}"

    out_path = plot_dmp_forcing_fit_from_trial(
        trial_dir=trial_dir,
        out_path=args.out,
        joint_idx=int(args.joint),
        n_basis=int(args.n_basis),
        angles_source=args.angles_source,
        derivative_method=args.derivative_method,
        savgol_window_length=int(args.savgol_window_length),
        savgol_polyorder=int(args.savgol_polyorder),
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

