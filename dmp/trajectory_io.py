from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from kinematics.joint_dynamics import smooth_angles_deg
from vis.trial_naming import trial_prefix

from .dmp import fit, rollout_simple


AnglesSource = Literal["auto", "raw", "clean"]


def load_angles_demo(trial_dir: Path, source: AnglesSource = "auto") -> np.ndarray:
    """Load elbow + shoulder angles from a trial directory; return (T, 4) demo in radians.

    Convention:
        column 0: elbow flexion
        column 1: shoulder flexion
        column 2: shoulder abduction
        column 3: shoulder internal rotation

    Prefer radians (elbow_rad, shoulder_rad); fall back to degrees if needed.
    """
    if source not in {"auto", "raw", "clean"}:
        raise ValueError(f"Invalid source '{source}', expected one of 'auto', 'raw', 'clean'.")

    prefix = trial_prefix(trial_dir)

    def _first_existing(paths: list[Path]) -> Path | None:
        for p in paths:
            if p.exists():
                return p
        return None

    if source == "raw":
        npz_path = _first_existing([trial_dir / f"{prefix}angles_raw.npz", trial_dir / "angles_raw.npz"])
    elif source == "clean":
        npz_path = _first_existing([trial_dir / f"{prefix}angles_clean.npz", trial_dir / "angles_clean.npz"])
    else:
        # auto: prefer canonical angles.npz, else fall back to raw/clean
        npz_path = _first_existing([trial_dir / f"{prefix}angles.npz", trial_dir / "angles.npz"])
        if npz_path is None:
            npz_path = _first_existing(
                [
                    trial_dir / f"{prefix}angles_raw.npz",
                    trial_dir / f"{prefix}angles_clean.npz",
                    trial_dir / "angles_raw.npz",
                    trial_dir / "angles_clean.npz",
                ]
            )

    if npz_path is None:
        raise FileNotFoundError(
            f"Could not find angles file in {trial_dir}. "
            "Expected one of 'angles*.npz' (optionally prefixed). "
            "Run mapping/sequence_to_angles.py first."
        )

    data = np.load(npz_path)
    if "elbow_rad" in data and "shoulder_rad" in data:
        elbow = data["elbow_rad"]
        shoulder = data["shoulder_rad"]
    else:
        elbow_deg = data["elbow_deg"]
        shoulder_deg = data["shoulder_deg"]
        elbow = np.deg2rad(elbow_deg)
        shoulder = np.deg2rad(shoulder_deg)

    if shoulder.ndim == 1:
        shoulder = shoulder[:, None]
    q_demo = np.column_stack([elbow, shoulder])

    valid = np.all(np.isfinite(q_demo), axis=1)
    q_demo = q_demo[valid]
    if q_demo.shape[0] < 10:
        raise ValueError(f"Not enough valid samples after cleaning: {q_demo.shape}")
    return q_demo


def load_dmp_trajectory(
    trial_dir: Path,
    *,
    prefer_saved_rollout: bool = True,
    rollout_source: Literal["clean", "raw"] = "clean",
    basis_functions: int | None = None,
    filter_order: int | None = None,
    n_basis: int = 15,
    alpha_canonical: float = 4.0,
    alpha_transformation: float = 25.0,
    beta_transformation: float = 6.25,
) -> tuple[np.ndarray, float]:
    """Load a DMP rollout trajectory for a trial, or fit+rollout from angles.

    If `prefer_saved_rollout` is True, this will first look for:
      - <prefix>dmp_rollout_<rollout_source>.npz
      - dmp_rollout_<rollout_source>.npz
    (and also tries the opposite source as a fallback).

    Returns:
      q_gen_rad: (T, 4) rollout in radians
      dt:        timestep used for the rollout
    """
    prefix = trial_prefix(trial_dir)

    def _load_rollout_npz(pth: Path) -> tuple[np.ndarray, float]:
        data = np.load(pth)
        if "q_gen_rad" not in data or "dt" not in data:
            raise KeyError(f"Unexpected contents in {pth.name}: keys={list(data.keys())}")
        q_gen = np.asarray(data["q_gen_rad"], dtype=float)
        dt = float(np.atleast_1d(data["dt"])[0])
        return q_gen, dt

    def _candidate_rollout_paths() -> list[Path]:
        # If the caller specifies sweep parameters, prefer the exact sweep filename produced by
        # run_full_pipeline.py:
        #   dmp_rollout_raw_n{n_basis}.npz
        #   dmp_rollout_clean_o{order}_n{n_basis}.npz
        paths: list[Path] = []
        if basis_functions is not None:
            if rollout_source == "clean":
                if filter_order is None:
                    raise ValueError(
                        "filter_order is required when rollout_source='clean' and basis_functions is set."
                    )
                paths.extend(
                    [
                        trial_dir / f"{prefix}dmp_rollout_clean_o{int(filter_order)}_n{int(basis_functions)}.npz",
                        trial_dir / f"dmp_rollout_clean_o{int(filter_order)}_n{int(basis_functions)}.npz",
                    ]
                )
            else:
                paths.extend(
                    [
                        trial_dir / f"{prefix}dmp_rollout_raw_n{int(basis_functions)}.npz",
                        trial_dir / f"dmp_rollout_raw_n{int(basis_functions)}.npz",
                    ]
                )

        ordered_sources = [rollout_source, "raw" if rollout_source == "clean" else "clean"]
        for src in ordered_sources:
            paths.extend(
                [
                    trial_dir / f"{prefix}dmp_rollout_{src}.npz",
                    trial_dir / f"dmp_rollout_{src}.npz",
                ]
            )
        return paths

    if prefer_saved_rollout:
        for pth in _candidate_rollout_paths():
            if pth.exists():
                return _load_rollout_npz(pth)

    q_demo = load_angles_demo(trial_dir, source="auto")
    #q_demo = np.deg2rad(smooth_angles_deg(np.degrees(q_demo)))

    T, n = q_demo.shape
    if n != 4:
        raise ValueError(f"Expected 4-DOF demo, got shape {q_demo.shape}")

    tau = 1.0
    dt = tau / (T - 1)

    model = fit(
        [q_demo],
        tau=tau,
        dt=dt,
        n_basis_functions=int(n_basis),
        alpha_canonical=float(alpha_canonical),
        alpha_transformation=float(alpha_transformation),
        beta_transformation=float(beta_transformation),
    )
    q_gen = rollout_simple(model, q_demo[0], q_demo[-1], tau=tau, dt=dt)
    return q_gen, float(dt)


def resolve_saved_dmp_rollout_path(
    trial_dir: Path,
    *,
    rollout_source: Literal["clean", "raw"] = "clean",
    basis_functions: int | None = None,
    filter_order: int | None = None,
) -> Path | None:
    """Return the first matching saved rollout path, or None if not found.

    This mirrors the selection logic used by `load_dmp_trajectory()` when
    `prefer_saved_rollout=True`.
    """
    prefix = trial_prefix(trial_dir)
    candidates: list[Path] = []

    if basis_functions is not None:
        if rollout_source == "clean":
            if filter_order is None:
                raise ValueError(
                    "filter_order is required when rollout_source='clean' and basis_functions is set."
                )
            candidates.extend(
                [
                    trial_dir / f"{prefix}dmp_rollout_clean_o{int(filter_order)}_n{int(basis_functions)}.npz",
                    trial_dir / f"dmp_rollout_clean_o{int(filter_order)}_n{int(basis_functions)}.npz",
                ]
            )
        else:
            candidates.extend(
                [
                    trial_dir / f"{prefix}dmp_rollout_raw_n{int(basis_functions)}.npz",
                    trial_dir / f"dmp_rollout_raw_n{int(basis_functions)}.npz",
                ]
            )

    ordered_sources = [rollout_source, "raw" if rollout_source == "clean" else "clean"]
    for src in ordered_sources:
        candidates.extend(
            [
                trial_dir / f"{prefix}dmp_rollout_{src}.npz",
                trial_dir / f"dmp_rollout_{src}.npz",
            ]
        )

    for p in candidates:
        if p.exists():
            return p
    return None

