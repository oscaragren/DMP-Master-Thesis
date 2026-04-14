"""
Exhaustive table-collision sweep for the standalone left arm in PyBullet.

This script mirrors the setup in `sim/limb_sim_table.py` (arm URDF + table pose),
but *does not* disable collisions between the arm and the table. It then checks
all integer-degree joint configurations within the configured limits and writes
all arm–table-colliding configurations to an output file.

Run from repo root, e.g.:

    python -m tests.test_collisions --out tests/invalid_joint_configs_table.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

try:
    import pybullet as p
except Exception as e:  # pragma: no cover
    p = None
    _PYBULLET_IMPORT_ERROR = e
else:
    _PYBULLET_IMPORT_ERROR = None


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _sim_dir() -> Path:
    return _project_root() / "sim"


def _joint_index(body_uid: int, joint_name: str) -> int:
    num_joints = p.getNumJoints(body_uid)
    for i in range(num_joints):
        info = p.getJointInfo(body_uid, i)
        name = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(info[1])
        if name == joint_name:
            return int(i)
    raise KeyError(f"Joint not found in URDF: {joint_name}")


def _spawn_table(*, pos: list[float], half_extents: list[float], rgba: list[float]) -> int:
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
    return int(
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=pos,
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
        )
    )


def _link_world_z(body_uid: int, link_index: int) -> float:
    link_state = p.getLinkState(body_uid, link_index, computeForwardKinematics=True)
    pos = link_state[0]  # linkWorldPosition
    return float(pos[2])


def _iter_integer_degrees_in_limits(*, lo_deg: int, hi_deg: int) -> range:
    if hi_deg < lo_deg:
        return range(0)
    return range(int(lo_deg), int(hi_deg) + 1)


def _floor_deg_bounds_from_rad(*, lo_rad: float, hi_rad: float) -> tuple[int, int]:
    """
    Convert rad bounds to integer-degree bounds by flooring endpoints, then
    later filtering exact rad-in-bound if needed.
    """
    lo = math.floor(math.degrees(float(lo_rad)))
    hi = math.floor(math.degrees(float(hi_rad)))
    return int(lo), int(hi)


def _check_collision(
    *,
    robot: int,
    table_id: int,
    sh_rotz: int,
    sh_roty: int,
    sh_rotx: int,
    elbow_roty: int,
    elbow_rad: float,
    sh_flex_rad: float,
    sh_abd_mapped_rad: float,
    sh_int_rad: float,
) -> tuple[bool, int]:
    """
    Returns (collides, n_contacts) for the given configuration.
    """
    # Apply joints (same mapping as limb_sim_table playback).
    p.resetJointState(robot, sh_rotz, float(sh_abd_mapped_rad))
    p.resetJointState(robot, sh_roty, float(sh_flex_rad))
    p.resetJointState(robot, sh_rotx, float(sh_int_rad))
    p.resetJointState(robot, elbow_roty, float(elbow_rad))

    p.performCollisionDetection()
    contacts = p.getContactPoints(bodyA=robot, bodyB=table_id)
    n_contacts = int(len(contacts))
    return (n_contacts > 0), n_contacts


def _first_valid_elbow_index_monotone(
    *,
    elbow_rads: list[float],
    check_at_index,
) -> tuple[int, int]:
    """
    Given a monotone predicate over elbow index i:
      invalid(i) is True for low i and becomes False at/after some boundary,
    return (first_valid_index, evaluations_used).

    If all invalid -> returns (len(elbow_rads), evals)
    If all valid   -> returns (0, evals)
    """
    n = len(elbow_rads)
    if n == 0:
        return 0, 0

    evals = 0
    lo = 0
    hi = n - 1

    inv_lo = bool(check_at_index(lo))
    evals += 1
    if not inv_lo:
        return 0, evals  # all valid

    inv_hi = bool(check_at_index(hi))
    evals += 1
    if inv_hi:
        return n, evals  # all invalid

    # Find first index where invalid becomes False (first valid)
    left = lo
    right = hi
    while left + 1 < right:
        mid = (left + right) // 2
        inv_mid = bool(check_at_index(mid))
        evals += 1
        if inv_mid:
            left = mid
        else:
            right = mid
    return right, evals


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Exhaustively test arm–table collisions for all integer-degree joint configurations."
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_project_root() / "tests" / "invalid_joint_configs_table.csv",
        help="Output CSV path for colliding (invalid) configurations.",
    )
    ap.add_argument(
        "--meta-out",
        type=Path,
        default=_project_root() / "tests" / "invalid_joint_configs_table.meta.txt",
        help="Output metadata/stats text file path.",
    )
    ap.add_argument("--gui", action="store_true", help="Use PyBullet GUI (very slow). Default: DIRECT.")

    # Table placement (match sim/limb_sim_table.py defaults)
    ap.add_argument("--table-x", type=float, default=0.45, help="Table center X in world coords (m).")
    ap.add_argument(
        "--table-y",
        type=float,
        default=-0.0,
        help="Table center Y in world coords (m).",
    )
    ap.add_argument("--table-top-z", type=float, default=0.0, help="Tabletop world Z (m).")
    ap.add_argument("--table-length-x", type=float, default=0.9, help="Table length along X (m).")
    ap.add_argument("--table-depth-y", type=float, default=0.6, help="Table depth along Y (m).")
    ap.add_argument("--table-height-z", type=float, default=0.7, help="Table height (m).")
    ap.add_argument("--table-rgba", type=str, default="1.0,0.0,0.0,0.5", help="RGBA (comma-separated).")

    # Arm base adjustment (match sim/limb_sim_table.py default)
    ap.add_argument(
        "--elbow-world-z",
        type=float,
        default=0.07,
        help="Shift the arm base so the elbow joint ends up at this world Z height (m).",
    )

    # Same optional abduction mapping as playback (defaults are identity).
    ap.add_argument("--abd-offset-deg", type=float, default=0.0)
    ap.add_argument("--abd-sign", type=float, default=1.0)

    # Practical knobs
    ap.add_argument(
        "--progress-every",
        type=int,
        default=250_000,
        help="Print progress every N configurations (DIRECT mode).",
    )
    ap.add_argument(
        "--flush-every",
        type=int,
        default=10_000,
        help="Flush CSV to disk every N invalid rows.",
    )
    ap.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="If set, stop after checking this many configurations (debug/smoke).",
    )
    ap.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bar even if tqdm is installed.",
    )
    ap.add_argument(
        "--assume-elbow-monotone",
        action="store_true",
        default=True,
        help=(
            "Use the assumption: for fixed shoulder joints, if a given elbow angle collides with the table, "
            "then every *decrease* in elbow angle also collides. This enables boundary-search pruning. "
            "Enabled by default."
        ),
    )

    args = ap.parse_args()

    if p is None:  # pragma: no cover
        raise RuntimeError(
            "PyBullet is required to run this test script, but it is not installed in the Python interpreter "
            f"you're using.\n\nOriginal import error: {type(_PYBULLET_IMPORT_ERROR).__name__}: {_PYBULLET_IMPORT_ERROR}\n"
        )

    # Import joint limits / mapping from project code.
    sim_dir = _sim_dir()
    if str(_project_root()) not in sys.path:
        sys.path.insert(0, str(_project_root()))
    from sim.joint_limits import DMP_LIMIT_INDEX, JOINT_LIMITS_RAD  # noqa: E402

    limits_rad = np.asarray(JOINT_LIMITS_RAD, dtype=float)
    if limits_rad.shape != (4, 2):
        raise ValueError(f"Expected sim.joint_limits.JOINT_LIMITS_RAD shape (4,2); got {limits_rad.shape}")

    # DMP vector order is: [elbow, shoulder_flex, shoulder_abd, shoulder_int_rot]
    idx = DMP_LIMIT_INDEX
    elbow_lim = tuple(float(x) for x in limits_rad[idx.elbow])
    sh_flex_lim = tuple(float(x) for x in limits_rad[idx.shoulder_flex])
    sh_abd_lim = tuple(float(x) for x in limits_rad[idx.shoulder_abd])
    sh_int_lim = tuple(float(x) for x in limits_rad[idx.shoulder_int_rot])

    elbow_lo_deg, elbow_hi_deg = _floor_deg_bounds_from_rad(lo_rad=elbow_lim[0], hi_rad=elbow_lim[1])
    sh_flex_lo_deg, sh_flex_hi_deg = _floor_deg_bounds_from_rad(lo_rad=sh_flex_lim[0], hi_rad=sh_flex_lim[1])
    sh_abd_lo_deg, sh_abd_hi_deg = _floor_deg_bounds_from_rad(lo_rad=sh_abd_lim[0], hi_rad=sh_abd_lim[1])
    sh_int_lo_deg, sh_int_hi_deg = _floor_deg_bounds_from_rad(lo_rad=sh_int_lim[0], hi_rad=sh_int_lim[1])

    elbow_vals = _iter_integer_degrees_in_limits(lo_deg=elbow_lo_deg, hi_deg=elbow_hi_deg)
    sh_flex_vals = _iter_integer_degrees_in_limits(lo_deg=sh_flex_lo_deg, hi_deg=sh_flex_hi_deg)
    sh_abd_vals = _iter_integer_degrees_in_limits(lo_deg=sh_abd_lo_deg, hi_deg=sh_abd_hi_deg)
    sh_int_vals = _iter_integer_degrees_in_limits(lo_deg=sh_int_lo_deg, hi_deg=sh_int_hi_deg)

    elbow_degs = list(elbow_vals)
    sh_flex_degs = list(sh_flex_vals)
    sh_abd_degs = list(sh_abd_vals)
    sh_int_degs = list(sh_int_vals)

    total = int(len(elbow_degs) * len(sh_flex_degs) * len(sh_abd_degs) * len(sh_int_degs))

    # Connect PyBullet and load arm URDF with same base orientation as limb_sim_table.
    if bool(args.gui):
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)
    p.setGravity(0, 0, 0)
    p.setAdditionalSearchPath(str(sim_dir))

    urdf_rel = "arm/new_left_arm.urdf"
    urdf_path = sim_dir / urdf_rel
    if not urdf_path.exists():
        p.disconnect()
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    base_orn = tuple(float(v) for v in p.getQuaternionFromEuler([math.pi / 2.0, 0.0, math.pi / 2.0]))
    robot = int(
        p.loadURDF(
            urdf_rel,
            basePosition=[0, 0, 0],
            baseOrientation=base_orn,
            useFixedBase=True,
        )
    )

    # Disable motors so resetJointState fully controls pose.
    for j in range(p.getNumJoints(robot)):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, force=0.0)

    # Calibrate base Z so the elbow joint lands at the requested world Z (same idea as limb_sim_table).
    elbow_roty = _joint_index(robot, "jLeftElbow_roty")
    p.resetJointState(robot, elbow_roty, 0.0)
    elbow_z0 = _link_world_z(robot, elbow_roty)
    base_pos0, base_orn0 = p.getBasePositionAndOrientation(robot)
    dz = float(args.elbow_world_z) - float(elbow_z0)
    p.resetBasePositionAndOrientation(
        robot,
        [float(base_pos0[0]), float(base_pos0[1]), float(base_pos0[2]) + dz],
        base_orn0,
    )

    # Spawn table exactly like limb_sim_table (but DO NOT disable collisions).
    rgba_parts = [s.strip() for s in str(args.table_rgba).split(",") if s.strip() != ""]
    if len(rgba_parts) != 4:
        raise ValueError("--table-rgba must contain 4 comma-separated numbers.")
    table_rgba = [float(x) for x in rgba_parts]

    half_extents = [
        0.5 * float(args.table_length_x),
        0.5 * float(args.table_depth_y),
        0.5 * float(args.table_height_z),
    ]
    table_center_z = float(args.table_top_z) - float(half_extents[2])
    table_id = _spawn_table(
        pos=[float(args.table_x), float(args.table_y), float(table_center_z)],
        half_extents=half_extents,
        rgba=table_rgba,
    )

    # Resolve joint indices used by playback mapping.
    sh_rotz = _joint_index(robot, "jLeftShoulder_rotz")  # shoulder abduction (+offset/sign)
    sh_rotx = _joint_index(robot, "jLeftShoulder_rotx")  # shoulder internal rotation
    sh_roty = _joint_index(robot, "jLeftShoulder_roty")  # shoulder flexion

    abd_offset = math.radians(float(args.abd_offset_deg))
    abd_sign = float(args.abd_sign)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)

    invalid_rows_written = 0
    invalid_count = 0
    valid_count = 0
    checked = 0

    t0 = time.time()
    use_tqdm = (tqdm is not None) and (not bool(args.no_tqdm))
    total_for_bar = total if args.max_configs is None else min(total, int(args.max_configs))

    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "elbow_deg",
                "shoulder_flex_deg",
                "shoulder_abd_deg",
                "shoulder_int_rot_deg",
                "elbow_rad",
                "shoulder_flex_rad",
                "shoulder_abd_rad",
                "shoulder_int_rot_rad",
                "n_contacts",
                "collision_inferred",
            ]
        )

        pbar = tqdm(total=total_for_bar, unit="cfg", smoothing=0.05) if use_tqdm else None

        # Iterate shoulder combinations; handle elbow range in bulk per shoulder config.
        shoulder_combos = itertools.product(sh_flex_degs, sh_abd_degs, sh_int_degs)
        for sh_flex_deg, sh_abd_deg, sh_int_deg in shoulder_combos:
            if args.max_configs is not None and checked >= int(args.max_configs):
                break

            sh_flex_rad = math.radians(float(sh_flex_deg))
            if sh_flex_rad < sh_flex_lim[0] - 1e-12 or sh_flex_rad > sh_flex_lim[1] + 1e-12:
                continue
            sh_abd_rad = math.radians(float(sh_abd_deg))
            if sh_abd_rad < sh_abd_lim[0] - 1e-12 or sh_abd_rad > sh_abd_lim[1] + 1e-12:
                continue
            sh_int_rad = math.radians(float(sh_int_deg))
            if sh_int_rad < sh_int_lim[0] - 1e-12 or sh_int_rad > sh_int_lim[1] + 1e-12:
                continue

            sh_abd_mapped = abd_sign * sh_abd_rad + abd_offset

            # Determine how many elbow configs we are allowed to process for this shoulder triple
            # when --max-configs is set.
            elbow_degs_this = elbow_degs
            if args.max_configs is not None:
                remaining = int(args.max_configs) - checked
                if remaining <= 0:
                    break
                elbow_degs_this = elbow_degs[: min(len(elbow_degs), remaining)]

            elbow_rads_this = [math.radians(float(d)) for d in elbow_degs_this]

            if bool(args.assume_elbow_monotone):

                def invalid_at(i: int) -> bool:
                    collides, _n = _check_collision(
                        robot=robot,
                        table_id=table_id,
                        sh_rotz=sh_rotz,
                        sh_roty=sh_roty,
                        sh_rotx=sh_rotx,
                        elbow_roty=elbow_roty,
                        elbow_rad=elbow_rads_this[i],
                        sh_flex_rad=sh_flex_rad,
                        sh_abd_mapped_rad=sh_abd_mapped,
                        sh_int_rad=sh_int_rad,
                    )
                    return bool(collides)

                first_valid_idx, _evals = _first_valid_elbow_index_monotone(
                    elbow_rads=elbow_rads_this,
                    check_at_index=invalid_at,
                )

                # Indices [0, first_valid_idx) are invalid; [first_valid_idx, end) are valid.
                n_invalid_here = int(first_valid_idx)
                n_valid_here = int(len(elbow_degs_this) - first_valid_idx)

                # Write all invalid configurations without individually re-checking contacts.
                for i in range(n_invalid_here):
                    elbow_deg = elbow_degs_this[i]
                    elbow_rad = elbow_rads_this[i]
                    invalid_count += 1
                    w.writerow(
                        [
                            int(elbow_deg),
                            int(sh_flex_deg),
                            int(sh_abd_deg),
                            int(sh_int_deg),
                            float(elbow_rad),
                            float(sh_flex_rad),
                            float(sh_abd_rad),
                            float(sh_int_rad),
                            -1,
                            1,
                        ]
                    )
                    invalid_rows_written += 1
                    if (invalid_rows_written % int(args.flush_every)) == 0:
                        f.flush()

                valid_count += n_valid_here
                checked += len(elbow_degs_this)
            else:
                # Fallback: brute force all elbows for this shoulder triple.
                for elbow_deg, elbow_rad in zip(elbow_degs_this, elbow_rads_this, strict=True):
                    collides, n_contacts = _check_collision(
                        robot=robot,
                        table_id=table_id,
                        sh_rotz=sh_rotz,
                        sh_roty=sh_roty,
                        sh_rotx=sh_rotx,
                        elbow_roty=elbow_roty,
                        elbow_rad=elbow_rad,
                        sh_flex_rad=sh_flex_rad,
                        sh_abd_mapped_rad=sh_abd_mapped,
                        sh_int_rad=sh_int_rad,
                    )
                    checked += 1
                    if collides:
                        invalid_count += 1
                        w.writerow(
                            [
                                int(elbow_deg),
                                int(sh_flex_deg),
                                int(sh_abd_deg),
                                int(sh_int_deg),
                                float(elbow_rad),
                                float(sh_flex_rad),
                                float(sh_abd_rad),
                                float(sh_int_rad),
                                int(n_contacts),
                                0,
                            ]
                        )
                        invalid_rows_written += 1
                        if (invalid_rows_written % int(args.flush_every)) == 0:
                            f.flush()
                    else:
                        valid_count += 1

                    if args.max_configs is not None and checked >= int(args.max_configs):
                        break

            if pbar is not None:
                pbar.update(len(elbow_degs_this))
                pbar.set_postfix(invalid=invalid_count, refresh=False)
            elif (not bool(args.gui)) and int(args.progress_every) > 0 and (checked % int(args.progress_every) == 0):
                elapsed = max(1e-9, time.time() - t0)
                rate = float(checked) / elapsed
                inv_pct = 100.0 * float(invalid_count) / max(1, checked)
                print(
                    f"[{checked:>10d}/{total}] invalid={invalid_count} ({inv_pct:.2f}%)  "
                    f"rate={rate:,.0f} cfg/s  elapsed={elapsed/60.0:.1f} min"
                )

        if pbar is not None:
            pbar.close()

    elapsed = max(1e-9, time.time() - t0)
    invalid_pct = 100.0 * float(invalid_count) / max(1, checked)
    valid_pct = 100.0 * float(valid_count) / max(1, checked)

    meta = [
        "arm-table collision sweep",
        f"checked={checked}",
        f"valid={valid_count}",
        f"invalid={invalid_count}",
        f"valid_pct={valid_pct:.6f}",
        f"invalid_pct={invalid_pct:.6f}",
        f"elapsed_s={elapsed:.6f}",
        f"rate_cfg_per_s={float(checked)/elapsed:.3f}",
        f"out_csv={args.out}",
        "",
        "limits (DMP order: elbow, shoulder_flex, shoulder_abd, shoulder_int_rot)",
        f"  elbow_deg=[{elbow_lo_deg}, {elbow_hi_deg}]  rad=[{elbow_lim[0]:.6f}, {elbow_lim[1]:.6f}]",
        f"  shoulder_flex_deg=[{sh_flex_lo_deg}, {sh_flex_hi_deg}]  rad=[{sh_flex_lim[0]:.6f}, {sh_flex_lim[1]:.6f}]",
        f"  shoulder_abd_deg=[{sh_abd_lo_deg}, {sh_abd_hi_deg}]  rad=[{sh_abd_lim[0]:.6f}, {sh_abd_lim[1]:.6f}]",
        f"  shoulder_int_rot_deg=[{sh_int_lo_deg}, {sh_int_hi_deg}]  rad=[{sh_int_lim[0]:.6f}, {sh_int_lim[1]:.6f}]",
        "",
        "world setup (matching sim/limb_sim_table defaults unless overridden)",
        f"  elbow_world_z={float(args.elbow_world_z):.6f}",
        f"  table_center=({float(args.table_x):.6f},{float(args.table_y):.6f},{float(table_center_z):.6f})",
        f"  table_half_extents=({half_extents[0]:.6f},{half_extents[1]:.6f},{half_extents[2]:.6f})",
        f"  table_top_z={float(args.table_top_z):.6f}",
        f"  abd_sign={abd_sign:.6f}",
        f"  abd_offset_deg={float(args.abd_offset_deg):.6f}",
        "",
    ]
    args.meta_out.write_text("\n".join(meta), encoding="utf-8")

    print()
    print("Done.")
    print(f"- Checked: {checked:,} configs")
    print(f"- Invalid (collide): {invalid_count:,} ({invalid_pct:.3f}%)")
    print(f"- Valid (no collide): {valid_count:,} ({valid_pct:.3f}%)")
    print(f"- Wrote invalid CSV: {args.out}")
    print(f"- Wrote stats/meta: {args.meta_out}")

    p.disconnect()


if __name__ == "__main__":
    main()