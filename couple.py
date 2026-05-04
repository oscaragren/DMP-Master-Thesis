from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dmp.dmp import fit as fit_dmp


def _trial_index_from_name(name: str) -> int | None:
    # trial_006 -> 6
    if not name.startswith("trial_"):
        return None
    s = name.removeprefix("trial_")
    try:
        return int(s)
    except ValueError:
        return None


def average_subject_curvature_weights(
    subject_dir: Path,
    *,
    motion: str | None = None,
    trial_glob: str = "trial_*",
    trial_min: int | None = None,
    trial_max: int | None = None,
    in_name: str = "curvature_weights.npz",
    out_name: str = "curvature_weights_mean.npz",
) -> Path:
    subject_dir = subject_dir.resolve()
    if not subject_dir.exists():
        raise FileNotFoundError(subject_dir)

    weights_list: list[np.ndarray] = []
    shapes: set[tuple[int, ...]] = set()

    search_root = subject_dir / motion if motion is not None else subject_dir
    trial_dirs = sorted([p for p in search_root.glob(trial_glob) if p.is_dir()])
    if trial_min is not None or trial_max is not None:
        lo = -10**9 if trial_min is None else int(trial_min)
        hi = 10**9 if trial_max is None else int(trial_max)
        filtered: list[Path] = []
        for p in trial_dirs:
            idx = _trial_index_from_name(p.name)
            if idx is None:
                continue
            if lo <= idx <= hi:
                filtered.append(p)
        trial_dirs = filtered
    if not trial_dirs:
        raise FileNotFoundError(f"No trial directories found under {search_root} (glob={trial_glob!r})")

    for trial_dir in trial_dirs:
        # 1) Preferred: a saved curvature_weights.npz (produced by C_analyze_data.py).
        cw_path = trial_dir / in_name
        if cw_path.exists():
            data = np.load(cw_path, allow_pickle=False)
            if "curvature_weights" in data:
                w = np.asarray(data["curvature_weights"], dtype=float)
                if w.ndim == 2:
                    weights_list.append(w)
                    shapes.add(tuple(w.shape))
                    continue

        # 2) Fallback: recompute curvature weights from angles + DMP hyperparams.
        angles_path = trial_dir / "angles.npz"
        model_path = trial_dir / "dmp_model.npz"
        if not angles_path.exists() or not model_path.exists():
            continue

        angles_npz = np.load(angles_path, allow_pickle=False)
        if "angles" not in angles_npz:
            continue
        angles = np.asarray(angles_npz["angles"], dtype=float)

        model_npz = np.load(model_path, allow_pickle=False)
        n_basis = int(np.asarray(model_npz["weights"]).shape[1]) if "weights" in model_npz else 100
        alpha_canonical = float(np.atleast_1d(model_npz["alpha_canonical"])[0]) if "alpha_canonical" in model_npz else 4.0
        alpha_transformation = (
            float(np.atleast_1d(model_npz["alpha_transformation"])[0]) if "alpha_transformation" in model_npz else 25.0
        )
        beta_transformation = (
            float(np.atleast_1d(model_npz["beta_transformation"])[0]) if "beta_transformation" in model_npz else 6.25
        )

        dt = float(np.atleast_1d(angles_npz["dt"])[0]) if "dt" in angles_npz else 1.0 / (angles.shape[0] - 1)
        # Note: The repo pipeline fits DMPs directly on these stored angles (often in degrees),
        # so we keep the same units here for consistency.
        model = fit_dmp(
            [angles],
            tau=1.0,
            dt=dt,
            n_basis_functions=n_basis,
            alpha_canonical=alpha_canonical,
            alpha_transformation=alpha_transformation,
            beta_transformation=beta_transformation,
        )
        w = np.asarray(model.curvature_weights, dtype=float)
        if w.ndim == 2:
            weights_list.append(w)
            shapes.add(tuple(w.shape))

    if not weights_list:
        raise ValueError(
            f"No curvature weights could be loaded or computed under {search_root}. "
            f"Expected '{in_name}', or both 'angles.npz' and 'dmp_model.npz' per trial."
        )
    if len(shapes) != 1:
        raise ValueError(f"Curvature weight shapes differ across trials: {sorted(shapes)}")

    w_stack = np.stack(weights_list, axis=0)  # (N_trials, n_joints, n_basis)
    w_mean = np.mean(w_stack, axis=0)

    out_path = subject_dir / out_name if motion is None else (subject_dir / motion / out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, curvature_weights=w_mean, n_trials=len(weights_list))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average curvature weights across all trials for a subject and save a new set."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all subjects 1..10 and trials 1..8 (optionally within --motion).",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="subject_10",
        help="Subject folder name under data/processed (e.g. subject_10 or subject_01).",
    )
    parser.add_argument(
        "--motion",
        type=str,
        default=None,
        help="Optional motion folder name (e.g. move_cup). If set, only averages within that motion.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="Root processed data directory (default: data/processed).",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="curvature_weights_mean.npz",
        help="Output filename to write inside the subject folder (or motion folder if --motion is set).",
    )
    args = parser.parse_args()

    processed_root = args.processed_root.expanduser()

    if args.all:
        for s in range(1, 11):
            subject_name = f"subject_{s:02d}" if s < 10 else "subject_10"
            subject_dir = processed_root / subject_name
            try:
                out = average_subject_curvature_weights(
                    subject_dir,
                    motion=args.motion,
                    trial_min=1,
                    trial_max=8,
                    out_name=args.out_name,
                )
                print(f"Wrote: {out}")
            except Exception as e:
                print(f"Skipped {subject_name}: {e}")
        return

    subject_dir = (processed_root / args.subject)
    out = average_subject_curvature_weights(
        subject_dir,
        motion=args.motion,
        out_name=args.out_name,
    )
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()