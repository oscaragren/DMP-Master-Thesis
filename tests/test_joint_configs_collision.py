"""
Visualize colliding joint configurations in the PyBullet GUI.

Reads `tests/invalid_joint_configs_table.csv`, samples N random configurations,
and displays them one-by-one using the same arm+table setup as `sim/limb_sim_table.py`.

Run from repo root:

    python -m tests.test_joint_configs_collision
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import pybullet as p
except Exception as e:  # pragma: no cover
    p = None
    _PYBULLET_IMPORT_ERROR = e
else:
    _PYBULLET_IMPORT_ERROR = None


@dataclass(frozen=True)
class JointConfig:
    elbow_deg: int
    shoulder_flex_deg: int
    shoulder_abd_deg: int
    shoulder_int_rot_deg: int


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


def _load_invalid_configs(csv_path: Path) -> list[JointConfig]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows: list[JointConfig] = []
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            return []

        required = {"elbow_deg", "shoulder_flex_deg", "shoulder_abd_deg", "shoulder_int_rot_deg"}
        missing = required.difference(set(r.fieldnames))
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in r:
            rows.append(
                JointConfig(
                    elbow_deg=int(float(row["elbow_deg"])),
                    shoulder_flex_deg=int(float(row["shoulder_flex_deg"])),
                    shoulder_abd_deg=int(float(row["shoulder_abd_deg"])),
                    shoulder_int_rot_deg=int(float(row["shoulder_int_rot_deg"])),
                )
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample and visualize colliding joint configs in limb_sim_table setup.")
    ap.add_argument(
        "--csv",
        type=Path,
        default=_project_root() / "tests" / "invalid_joint_configs_table.csv",
        help="Path to invalid config CSV produced by tests/test_collisions.py",
    )
    ap.add_argument("--n", type=int, default=15, help="Number of random invalid configs to show.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")

    # Same world defaults as sim/limb_sim_table.py
    ap.add_argument("--table-x", type=float, default=0.45)
    ap.add_argument("--table-y", type=float, default=-0.0)
    ap.add_argument("--table-top-z", type=float, default=0.0)
    ap.add_argument("--table-length-x", type=float, default=0.9)
    ap.add_argument("--table-depth-y", type=float, default=0.6)
    ap.add_argument("--table-height-z", type=float, default=0.7)
    ap.add_argument("--table-rgba", type=str, default="1.0,0.0,0.0,0.5")
    ap.add_argument("--elbow-world-z", type=float, default=0.07)

    ap.add_argument("--abd-offset-deg", type=float, default=0.0)
    ap.add_argument("--abd-sign", type=float, default=1.0)

    ap.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        help="If >0, auto-advance after this many seconds instead of waiting for Enter.",
    )
    args = ap.parse_args()

    if p is None:  # pragma: no cover
        raise RuntimeError(
            "PyBullet is required to run this visualization script, but it is not installed in the Python "
            f"interpreter you're using.\n\nOriginal import error: {type(_PYBULLET_IMPORT_ERROR).__name__}: {_PYBULLET_IMPORT_ERROR}\n"
        )

    invalid = _load_invalid_configs(args.csv)
    if len(invalid) == 0:
        raise RuntimeError(
            f"No invalid configurations found in {args.csv}. "
            "Run `python -m tests.test_collisions` first to generate it."
        )

    rng = random.Random(int(args.seed))
    n = min(int(args.n), len(invalid))
    sample = rng.sample(invalid, k=n)

    # Ensure project root is on sys.path (mirrors sim scripts)
    if str(_project_root()) not in sys.path:
        sys.path.insert(0, str(_project_root()))

    # Load arm + table in GUI (same as limb_sim_table).
    p.connect(p.GUI)
    p.setGravity(0, 0, 0)
    p.setAdditionalSearchPath(str(_sim_dir()))

    urdf_rel = "arm/new_left_arm.urdf"
    urdf_path = _sim_dir() / urdf_rel
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

    for j in range(p.getNumJoints(robot)):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, force=0.0)

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

    sh_rotz = _joint_index(robot, "jLeftShoulder_rotz")
    sh_rotx = _joint_index(robot, "jLeftShoulder_rotx")
    sh_roty = _joint_index(robot, "jLeftShoulder_roty")

    abd_offset = math.radians(float(args.abd_offset_deg))
    abd_sign = float(args.abd_sign)

    print(f"Loaded {n} random invalid configurations from {args.csv}")
    print("Close the PyBullet window to exit.")

    try:
        for i, cfg in enumerate(sample, start=1):
            elbow = math.radians(float(cfg.elbow_deg))
            sh_flex = math.radians(float(cfg.shoulder_flex_deg))
            sh_abd = math.radians(float(cfg.shoulder_abd_deg))
            sh_int = math.radians(float(cfg.shoulder_int_rot_deg))
            sh_abd_mapped = abd_sign * sh_abd + abd_offset

            p.resetJointState(robot, sh_rotz, float(sh_abd_mapped))
            p.resetJointState(robot, sh_roty, float(sh_flex))
            p.resetJointState(robot, sh_rotx, float(sh_int))
            p.resetJointState(robot, elbow_roty, float(elbow))

            p.performCollisionDetection()
            contacts = p.getContactPoints(bodyA=robot, bodyB=table_id)

            print(
                f"[{i:02d}/{n:02d}] elbow={cfg.elbow_deg:>4d}  sh_flex={cfg.shoulder_flex_deg:>4d}  "
                f"sh_abd={cfg.shoulder_abd_deg:>4d}  sh_int={cfg.shoulder_int_rot_deg:>4d}  "
                f"contacts={len(contacts)}"
            )

            if float(args.hold_seconds) > 0.0:
                time.sleep(float(args.hold_seconds))
            else:
                input("Press Enter for next configuration...")
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
