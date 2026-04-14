"""
Generate valid joint-angle combinations given:
- joint limits (see sim/joint_limits.py)
- a simple table-clearance model based on elbow height above the table

Model / assumptions (conservative, geometry-only):
- Table top is a plane at z = table_z in world coordinates (default 0).
- "Elbow center" is approximated by the world position of the link after the
  elbow flexion joint in the standalone left-arm URDF (sim/arm/left_arm.urdf).
- Forearm "vertical half-height" below the elbow joint is forearm_length/2.
  If the elbow center is at height h_e above the tabletop, then a conservative
  no-hit condition for the forearm is:

    h_e - (forearm_length/2) - forearm_radius  >= safety_margin

Because absolute table pose vs robot base is unknown, we *calibrate* the base z
offset so that, at a chosen calibration joint configuration, the elbow center
is elbow_height_m above the table top.

Outputs:
- NPZ file with either:
  - q_valid (N,4), clearance_m (N,), elbow_z (N,)  (when --which valid)
  - q_invalid (N,4), clearance_m (N,), elbow_z (N,) (when --which invalid)

Run from repo root, e.g.:
  python sim/get_valid_angle_combinations.py --mode grid --grid-points 35
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

try:
    import pybullet as p
except Exception as e:  # pragma: no cover
    p = None
    _PYBULLET_IMPORT_ERROR = e
else:
    _PYBULLET_IMPORT_ERROR = None

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _sim_dir() -> Path:
    return Path(__file__).resolve().parent


def joint_index(body_uid: int, joint_name: str) -> int:
    """Resolve a joint name to its PyBullet index."""
    if p is None:  # pragma: no cover
        raise RuntimeError("PyBullet is not available in this Python environment.")
    num_joints = p.getNumJoints(body_uid)
    for i in range(num_joints):
        info = p.getJointInfo(body_uid, i)
        name = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(info[1])
        if name == joint_name:
            return i
    raise KeyError(f"Joint not found in URDF: {joint_name}")


def _load_left_arm_urdf(gui: bool) -> int:
    """Load standalone left arm URDF used by sim/simulation.py."""
    if p is None:  # pragma: no cover
        raise RuntimeError("PyBullet is not available in this Python environment.")
    if gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    # URDF uses "arm/...", so search path must be sim/ so that arm/ resolves.
    sim_dir = _sim_dir()
    p.setAdditionalSearchPath(str(sim_dir))
    p.setGravity(0, 0, 0)

    urdf_path = sim_dir / "arm" / "left_arm.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    body_uid = p.loadURDF(
        "arm/left_arm.urdf",
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
        useFixedBase=True,
    )

    # Disable motors so resetJointState has immediate effect.
    for j in range(p.getNumJoints(body_uid)):
        p.setJointMotorControl2(body_uid, j, p.POSITION_CONTROL, force=0.0)

    return body_uid


def _arm_joint_indices(body_uid: int) -> dict[str, int]:
    """
    Mapping between our 4-DOF vector and the standalone arm URDF joints.

    We follow sim/simulation.py controlled_names:
      - elbow flexion            -> jLeftElbow_roty
      - shoulder left/right      -> jLeftShoulder_rotz
      - shoulder up/down         -> jLeftShoulder_roty
      - upper arm rotation       -> jLeftShoulder_rotx
    """
    names = {
        "elbow": "jLeftElbow_roty",
        "shoulder_lr": "jLeftShoulder_rotz",
        "shoulder_ud": "jLeftShoulder_roty",
        "upper_arm_rot": "jLeftShoulder_rotx",
    }
    return {k: joint_index(body_uid, v) for k, v in names.items()}


def _set_q(body_uid: int, idx: dict[str, int], q: np.ndarray) -> None:
    """
    q order must match sim/joint_limits.py rows:
      0 shoulder up/down
      1 shoulder left/right
      2 elbow
      3 upper arm rotation
    """
    q0, q1, q2, q3 = (float(x) for x in q)
    p.resetJointState(body_uid, idx["shoulder_ud"], q0, targetVelocity=0.0)
    p.resetJointState(body_uid, idx["shoulder_lr"], q1, targetVelocity=0.0)
    p.resetJointState(body_uid, idx["elbow"], q2, targetVelocity=0.0)
    p.resetJointState(body_uid, idx["upper_arm_rot"], q3, targetVelocity=0.0)


def _elbow_world_z(body_uid: int, elbow_joint_index: int) -> float:
    """
    Approximate elbow center height as the world Z of the elbow's child link frame.
    In Bullet, link index == joint index.
    """
    p.performCollisionDetection()
    link_state = p.getLinkState(body_uid, elbow_joint_index, computeForwardKinematics=True)
    pos = link_state[0]  # linkWorldPosition
    return float(pos[2])


def _calibrate_base_z(
    body_uid: int,
    idx: dict[str, int],
    q_calib: np.ndarray,
    elbow_height_m: float,
    table_z: float,
) -> float:
    """
    Choose base z such that elbow is elbow_height_m above tabletop in q_calib.
    Returns the base_z used.
    """
    _set_q(body_uid, idx, q_calib)
    elbow_z0 = _elbow_world_z(body_uid, idx["elbow"])
    desired_elbow_z = float(table_z + elbow_height_m)
    base_z = desired_elbow_z - elbow_z0
    p.resetBasePositionAndOrientation(body_uid, [0.0, 0.0, base_z], [0.0, 0.0, 0.0, 1.0])
    return float(base_z)


def _iter_grid(limits: np.ndarray, grid_points: int) -> Iterable[np.ndarray]:
    grids = [
        np.linspace(float(lo), float(hi), int(grid_points), dtype=float)
        for lo, hi in limits
    ]
    for tup in itertools.product(*grids):
        yield np.asarray(tup, dtype=float)


def _iter_degree_step_grid(
    limits_rad: np.ndarray,
    step_deg: int,
    *,
    floor_limits: bool = True,
    include_endpoints: bool = True,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """
    Iterate a grid in *integer degrees* with the given step.

    Returns tuples (q_rad, q_deg) where q_deg is integer degrees.

    Rounding policy:
    - Convert rad limits to degrees.
    - If floor_limits=True, floor both ends in degrees.
    - Always filter so that the resulting q_rad is within the original radian limits.
    """
    if int(step_deg) <= 0:
        raise ValueError("step_deg must be a positive integer.")

    grids_deg: list[np.ndarray] = []
    for lo_rad, hi_rad in limits_rad:
        lo_deg_f = math.degrees(float(lo_rad))
        hi_deg_f = math.degrees(float(hi_rad))
        if floor_limits:
            lo_deg = math.floor(lo_deg_f)
            hi_deg = math.floor(hi_deg_f)
        else:
            lo_deg = int(round(lo_deg_f))
            hi_deg = int(round(hi_deg_f))

        if include_endpoints:
            vals = np.arange(lo_deg, hi_deg + 1, int(step_deg), dtype=int)
        else:
            vals = np.arange(lo_deg + int(step_deg), hi_deg, int(step_deg), dtype=int)
        grids_deg.append(vals)

    # Cartesian product over integer-degree grids
    lo_rad_all = limits_rad[:, 0]
    hi_rad_all = limits_rad[:, 1]
    for tup in itertools.product(*grids_deg):
        q_deg = np.asarray(tup, dtype=int)
        q_rad = np.deg2rad(q_deg.astype(float))

        # Filter to ensure we stay inside the original radian bounds.
        if np.any(q_rad < lo_rad_all - 1e-12) or np.any(q_rad > hi_rad_all + 1e-12):
            continue
        yield q_rad, q_deg


def _iter_random(limits: np.ndarray, n: int, seed: int) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    lo = limits[:, 0]
    hi = limits[:, 1]
    for _ in range(int(n)):
        u = rng.random(limits.shape[0], dtype=float)
        yield lo + (hi - lo) * u


def _compute_clearance(
    elbow_z: float,
    table_z: float,
    forearm_length_m: float,
    forearm_radius_m: float,
    safety_margin_m: float,
) -> float:
    h_e = float(elbow_z - table_z)
    return h_e - 0.5 * float(forearm_length_m) - float(forearm_radius_m) - float(safety_margin_m)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate table-safe joint configurations for the left arm.")
    parser.add_argument("--mode", choices=["grid", "random"], default="grid")
    parser.add_argument("--grid-points", type=int, default=35, help="Points per joint for grid mode.")
    parser.add_argument("--random-n", type=int, default=200_000, help="Number of random samples for random mode.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random mode.")
    parser.add_argument(
        "--which",
        choices=["valid", "invalid"],
        default="invalid",
        help="Which configurations to write. 'valid' uses clearance >= 0, 'invalid' uses clearance < 0.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_project_root() / "sim" / "invalid_angle_combinations.npz",
        help="Output .npz path.",
    )
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI (slow).")
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bar even if tqdm is installed.",
    )
    parser.add_argument(
        "--step-deg",
        type=int,
        default=None,
        help=(
            "If set, uses an integer-degree grid with this step (e.g. 1 => 0°,1°,2°,...). "
            "This overrides --mode/--grid-points. Endpoints are included; degree endpoints are floored, "
            "then filtered to remain within radian limits."
        ),
    )

    # Measurement / clearance model
    parser.add_argument("--table-z", type=float, default=0.0, help="Tabletop world z (m).")
    parser.add_argument("--elbow-height-m", type=float, default=0.07, help="Measured elbow height above table (m) in calibration pose.")
    parser.add_argument("--forearm-length-m", type=float, default=0.10, help="Forearm length (m).")
    parser.add_argument("--forearm-radius-m", type=float, default=0.0, help="Approx forearm radius/thickness (m).")
    parser.add_argument("--safety-margin-m", type=float, default=0.01, help="Extra clearance margin above table (m).")

    # Calibration pose (q in our 4-DOF order)
    parser.add_argument(
        "--calib-q-deg",
        type=str,
        default="0,0,0,0",
        help="Calibration pose joint angles in degrees: shoulder_ud, shoulder_lr, elbow, upper_arm_rot",
    )

    args = parser.parse_args()

    if p is None:  # pragma: no cover
        raise RuntimeError(
            "PyBullet is required to run this script, but it is not installed in the Python interpreter "
            f"you're using.\n\nOriginal import error: {type(_PYBULLET_IMPORT_ERROR).__name__}: {_PYBULLET_IMPORT_ERROR}\n\n"
            "Fix: activate the environment where you installed `requirements.txt` (which includes `pybullet`), "
            "or install it for this interpreter (e.g. `python3 -m pip install -r requirements.txt`)."
        )

    # Import limits from the project.
    from joint_limits import JOINT_LIMITS_RAD

    limits = np.asarray(JOINT_LIMITS_RAD, dtype=float)
    if limits.shape != (4, 2):
        raise ValueError(f"Expected JOINT_LIMITS_RAD shape (4,2), got {limits.shape}")

    # Parse calibration q
    parts = [p.strip() for p in str(args.calib_q_deg).split(",") if p.strip() != ""]
    if len(parts) != 4:
        raise ValueError("--calib-q-deg must contain 4 comma-separated numbers.")
    q_calib_deg = np.array([float(x) for x in parts], dtype=float)
    q_calib = np.deg2rad(q_calib_deg)

    # Clamp calibration q into limits (avoid invalid URDF range issues)
    q_calib = np.clip(q_calib, limits[:, 0], limits[:, 1])

    body_uid = _load_left_arm_urdf(gui=bool(args.gui))
    idx = _arm_joint_indices(body_uid)

    base_z = _calibrate_base_z(
        body_uid=body_uid,
        idx=idx,
        q_calib=q_calib,
        elbow_height_m=float(args.elbow_height_m),
        table_z=float(args.table_z),
    )

    if args.mode == "grid":
        it = _iter_grid(limits, int(args.grid_points))
        total = int(args.grid_points) ** 4
    else:
        it = _iter_random(limits, int(args.random_n), int(args.seed))
        total = int(args.random_n)

    keep_valid = str(args.which) == "valid"
    q_keep = []
    q_keep_deg = []
    clearance = []
    elbow_zs = []
    use_tqdm = (tqdm is not None) and (not bool(args.no_tqdm))

    if args.step_deg is not None:
        deg_it = _iter_degree_step_grid(limits, int(args.step_deg), floor_limits=True, include_endpoints=True)
        # total unknown without materializing; estimate by product of per-joint counts after flooring.
        counts = []
        for lo_rad, hi_rad in limits:
            lo_deg = math.floor(math.degrees(float(lo_rad)))
            hi_deg = math.floor(math.degrees(float(hi_rad)))
            counts.append(max(0, 1 + (hi_deg - lo_deg) // int(args.step_deg)))
        total = int(np.prod(np.asarray(counts, dtype=int)))

        if use_tqdm:
            deg_it_wrapped = tqdm(deg_it, total=total, unit="cfg", smoothing=0.05)
            for q, q_deg in deg_it_wrapped:
                _set_q(body_uid, idx, q)
                ez = _elbow_world_z(body_uid, idx["elbow"])
                c = _compute_clearance(
                    elbow_z=ez,
                    table_z=float(args.table_z),
                    forearm_length_m=float(args.forearm_length_m),
                    forearm_radius_m=float(args.forearm_radius_m),
                    safety_margin_m=float(args.safety_margin_m),
                )
                is_valid = c >= 0.0
                if (is_valid and keep_valid) or ((not is_valid) and (not keep_valid)):
                    q_keep.append(q)
                    q_keep_deg.append(q_deg)
                    clearance.append(c)
                    elbow_zs.append(ez)
                deg_it_wrapped.set_postfix(kept=len(q_keep), refresh=False)
        else:
            progress_every = max(1, total // 20) if total > 0 else 1
            for i, (q, q_deg) in enumerate(deg_it, start=1):
                _set_q(body_uid, idx, q)
                ez = _elbow_world_z(body_uid, idx["elbow"])
                c = _compute_clearance(
                    elbow_z=ez,
                    table_z=float(args.table_z),
                    forearm_length_m=float(args.forearm_length_m),
                    forearm_radius_m=float(args.forearm_radius_m),
                    safety_margin_m=float(args.safety_margin_m),
                )
                is_valid = c >= 0.0
                if (is_valid and keep_valid) or ((not is_valid) and (not keep_valid)):
                    q_keep.append(q)
                    q_keep_deg.append(q_deg)
                    clearance.append(c)
                    elbow_zs.append(ez)

                if (i % progress_every) == 0 or i == total:
                    print(f"[{i:>9d}/{total}] kept={len(q_keep)} ({(100.0*len(q_keep)/max(1,i)):.2f}%)")
    else:
        if use_tqdm:
            it_wrapped = tqdm(it, total=total, unit="cfg", smoothing=0.05)
            for q in it_wrapped:
                _set_q(body_uid, idx, q)
                ez = _elbow_world_z(body_uid, idx["elbow"])
                c = _compute_clearance(
                    elbow_z=ez,
                    table_z=float(args.table_z),
                    forearm_length_m=float(args.forearm_length_m),
                    forearm_radius_m=float(args.forearm_radius_m),
                    safety_margin_m=float(args.safety_margin_m),
                )
                is_valid = c >= 0.0
                if (is_valid and keep_valid) or ((not is_valid) and (not keep_valid)):
                    q_keep.append(q)
                    clearance.append(c)
                    elbow_zs.append(ez)
                it_wrapped.set_postfix(kept=len(q_keep), refresh=False)
        else:
            progress_every = max(1, total // 20)
            for i, q in enumerate(it, start=1):
                _set_q(body_uid, idx, q)
                ez = _elbow_world_z(body_uid, idx["elbow"])
                c = _compute_clearance(
                    elbow_z=ez,
                    table_z=float(args.table_z),
                    forearm_length_m=float(args.forearm_length_m),
                    forearm_radius_m=float(args.forearm_radius_m),
                    safety_margin_m=float(args.safety_margin_m),
                )
                is_valid = c >= 0.0
                if (is_valid and keep_valid) or ((not is_valid) and (not keep_valid)):
                    q_keep.append(q)
                    clearance.append(c)
                    elbow_zs.append(ez)

                if (i % progress_every) == 0 or i == total:
                    # Avoid importing tqdm; keep it simple.
                    print(f"[{i:>9d}/{total}] kept={len(q_keep)} ({(100.0*len(q_keep)/max(1,i)):.2f}%)")

    q_keep_arr = np.asarray(q_keep, dtype=float) if q_keep else np.zeros((0, 4), dtype=float)
    q_keep_deg_arr = np.asarray(q_keep_deg, dtype=int) if q_keep_deg else np.zeros((0, 4), dtype=int)
    clearance_arr = np.asarray(clearance, dtype=float) if clearance else np.zeros((0,), dtype=float)
    elbow_z_arr = np.asarray(elbow_zs, dtype=float) if elbow_zs else np.zeros((0,), dtype=float)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    key = "q_valid" if keep_valid else "q_invalid"
    key_deg = "q_valid_deg" if keep_valid else "q_invalid_deg"
    np.savez(
        args.out,
        **{key: q_keep_arr, key_deg: q_keep_deg_arr},
        clearance_m=clearance_arr,
        elbow_z=elbow_z_arr,
        limits_rad=limits,
        mode=str(args.mode),
        grid_points=int(args.grid_points),
        random_n=int(args.random_n),
        seed=int(args.seed),
        step_deg=(-1 if args.step_deg is None else int(args.step_deg)),
        table_z=float(args.table_z),
        elbow_height_m=float(args.elbow_height_m),
        forearm_length_m=float(args.forearm_length_m),
        forearm_radius_m=float(args.forearm_radius_m),
        safety_margin_m=float(args.safety_margin_m),
        calib_q_rad=q_calib,
        base_z=base_z,
    )

    print()
    print("Done.")
    print(f"- Wrote: {args.out}")
    print(f"- Total checked: {total}")
    print(f"- Kept ({args.which}): {q_keep_arr.shape[0]}")
    if q_keep_arr.shape[0] > 0:
        print(
            f"- Clearance (m): min={clearance_arr.min():.4f}, median={np.median(clearance_arr):.4f}, max={clearance_arr.max():.4f}"
        )
    else:
        print(f"- No {args.which} configurations found with current settings.")

    p.disconnect()


if __name__ == "__main__":
    main()

