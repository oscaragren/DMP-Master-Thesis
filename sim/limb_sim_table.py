"""
Play back a DMP‑generated left‑arm trajectory on the standalone arm model in PyBullet,
with a simple table placed in front of the body.

This mirrors sim/limb_sim.py, but additionally spawns a static table (a box) whose
top surface lies at `--table-top-z` in world coordinates.

Usage (from project root):

    python3 sim/limb_sim_table.py --subject 1 --motion lift --trial 6

or:

    python3 sim/limb_sim_table.py --path test_data/processed/subject_01/lift/trial_006
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import pybullet as p

# Ensure project root is on sys.path for imports
_sim_dir = Path(__file__).resolve().parent
_project_root = _sim_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dmp.trajectory_io import load_dmp_trajectory, resolve_saved_dmp_rollout_path
from sim.joint_limits import clamp_dmp_vector


def joint_index(body_uid: int, joint_name: str) -> int:
    """Resolve a joint name to its PyBullet index."""
    num_joints = p.getNumJoints(body_uid)
    for i in range(num_joints):
        info = p.getJointInfo(body_uid, i)
        name = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(info[1])
        if name == joint_name:
            return i
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

def _disable_collisions_between(body_a: int, body_b: int) -> None:
    """
    Disable collisions between all links of body_a and the base link of body_b.
    (body_b is the table, created as a single-link multibody where the base link index is -1.)
    """
    for link_a in range(-1, p.getNumJoints(body_a)):
        p.setCollisionFilterPair(body_a, body_b, link_a, -1, enableCollision=0)


def _link_world_z(body_uid: int, link_index: int) -> float:
    link_state = p.getLinkState(body_uid, link_index, computeForwardKinematics=True)
    pos = link_state[0]  # linkWorldPosition
    return float(pos[2])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play back a DMP-generated left-arm trajectory on the standalone arm URDF in PyBullet (with a table)."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to trial dir (overrides subject/motion/trial).",
    )
    parser.add_argument("--subject", type=int, default=1, help="Subject number")
    parser.add_argument("--motion", type=str, default="lift", help="Motion name (e.g. reach, lift)")
    parser.add_argument("--trial", type=int, default=6, help="Trial number")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_project_root / "test_data" / "processed",
        help="Root directory for processed data (subject/motion/trial underneath).",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["raw", "clean"],
        default="clean",
        help="Which DMP rollout to use (default: clean).",
    )
    parser.add_argument(
        "--n-basis",
        type=int,
        default=30,
        help="Basis function count to load (default: 30).",
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=2,
        help="For clean sweep rollouts: Butterworth filter order (default: 2).",
    )
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    parser.add_argument(
        "--abd-offset-deg",
        type=float,
        default=0.0,
        help="Constant offset applied to shoulder abduction before mapping to shoulder yaw joint (deg).",
    )
    parser.add_argument(
        "--abd-sign",
        type=float,
        default=1.0,
        help="Sign applied to shoulder abduction before offset (use -1 if direction is flipped).",
    )

    # Table placement (defaults: table centered in front of the arm, top at z=0)
    parser.add_argument("--table-x", type=float, default=0.45, help="Table center X in world coords (m).")
    parser.add_argument(
        "--table-y",
        type=float,
        default=-0.0,
        help="Table center Y in world coords (m). (Default is negative because +Y ends up behind the arm with current base orientation.)",
    )
    parser.add_argument("--table-top-z", type=float, default=0.0, help="Tabletop world Z (m).")
    parser.add_argument("--table-length-x", type=float, default=0.9, help="Table length along X (m).")
    parser.add_argument("--table-depth-y", type=float, default=0.6, help="Table depth along Y (m).")
    parser.add_argument("--table-height-z", type=float, default=0.7, help="Table height (m).")
    parser.add_argument(
        "--table-rgba",
        type=str,
        default="1.0,0.0,0.0,0.5",
        help="RGBA color for the table visual (comma-separated floats).",
    )
    parser.add_argument(
        "--elbow-world-z",
        type=float,
        default=0.07,
        help="Shift the arm base so the elbow joint ends up at this world Z height (m).",
    )

    args = parser.parse_args()

    if args.path is not None:
        trial_dir: Path = args.path
    else:
        trial_dir = (
            args.data_dir / f"subject_{args.subject:02d}" / args.motion / f"trial_{args.trial:03d}"
        )
    if not trial_dir.exists():
        raise FileNotFoundError(f"Trial directory not found: {trial_dir}")

    rollout_path = resolve_saved_dmp_rollout_path(
        trial_dir,
        rollout_source=args.source,
        basis_functions=args.n_basis,
        filter_order=args.filter_order,
    )
    if rollout_path is not None:
        print(f"Loading saved DMP rollout: {rollout_path}")
    else:
        print("No saved DMP rollout found; will fit+rollout from angles*.npz at runtime.")

    # 1. Load DMP trajectory (elbow + 3 shoulder DOFs)
    q_traj, dt = load_dmp_trajectory(
        trial_dir,
        rollout_source=args.source,
        basis_functions=args.n_basis,
        filter_order=args.filter_order,
    )  # (T, 4), radians
    q_traj = clamp_dmp_vector(q_traj)  # Clamp to joint limits

    T, _n = q_traj.shape
    traj_duration_s = float((T - 1) * dt)
    print(f"Trajectory duration (simulated): {traj_duration_s:.3f} s  (T={T}, dt={dt:.6f})")

    # 2. Connect PyBullet and load standalone arm URDF
    p.connect(p.GUI)
    p.setGravity(0, 0, 0)  # kinematic playback

    p.setAdditionalSearchPath(str(_sim_dir))
    urdf_rel = "arm/new_left_arm.urdf"
    urdf_path = _sim_dir / urdf_rel
    if not urdf_path.exists():
        p.disconnect()
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # Same base orientation as sim/limb_sim.py (see that file for rationale).
    base_orn = tuple(float(v) for v in p.getQuaternionFromEuler([math.pi / 2.0, 0.0, math.pi / 2.0]))
    robot = p.loadURDF(
        urdf_rel,
        basePosition=[0, 0, 0],
        baseOrientation=base_orn,
        useFixedBase=True,
    )

    # Raise/lower the arm so the elbow joint is at a known height above ground.
    elbow_roty = joint_index(robot, "jLeftElbow_roty")
    p.resetJointState(robot, elbow_roty, 0.0)
    elbow_z0 = _link_world_z(robot, elbow_roty)
    base_pos0, base_orn0 = p.getBasePositionAndOrientation(robot)
    dz = float(args.elbow_world_z) - float(elbow_z0)
    p.resetBasePositionAndOrientation(
        robot,
        [float(base_pos0[0]), float(base_pos0[1]), float(base_pos0[2]) + dz],
        base_orn0,
    )

    # 2b. Spawn the table in front of the body (in this sim: +Y direction by default).
    rgba_parts = [s.strip() for s in str(args.table_rgba).split(",") if s.strip() != ""]
    if len(rgba_parts) != 4:
        raise ValueError("--table-rgba must contain 4 comma-separated numbers.")
    table_rgba = [float(x) for x in rgba_parts]

    half_extents = [
        0.5 * float(args.table_length_x),
        0.5 * float(args.table_depth_y),
        0.5 * float(args.table_height_z),
    ]
    table_center_z = float(args.table_top_z) - half_extents[2]
    table_id = _spawn_table(
        pos=[float(args.table_x), float(args.table_y), table_center_z],
        half_extents=half_extents,
        rgba=table_rgba,
    )
    _disable_collisions_between(robot, table_id)

    print("Loaded arm joints:")
    for j in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        print(j, info[1].decode("utf-8"))

    # 3. Map DMP joints to arm joints (same mapping as sim/limb_sim.py)
    sh_rotz = joint_index(robot, "jLeftShoulder_rotz")
    sh_rotx = joint_index(robot, "jLeftShoulder_rotx")
    sh_roty = joint_index(robot, "jLeftShoulder_roty")

    # Disable motors so resetJointState fully controls pose
    num_joints = p.getNumJoints(robot)
    for j in range(num_joints):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, force=0.0)

    print(f"Starting DMP playback on standalone arm from {trial_dir}")
    abd_offset = math.radians(float(args.abd_offset_deg))
    abd_sign = float(args.abd_sign)

    try:
        while True:
            for t_idx in range(T):
                if not p.isConnected():
                    return
                q_t = q_traj[t_idx]

                elbow = float(q_t[0])
                sh_flex = float(q_t[1])
                sh_abd = float(q_t[2])
                sh_int = float(q_t[3])

                sh_abd_mapped = abd_sign * sh_abd + abd_offset
                p.resetJointState(robot, sh_rotz, sh_abd_mapped)
                p.resetJointState(robot, sh_roty, sh_flex)
                p.resetJointState(robot, sh_rotx, sh_int)
                p.resetJointState(robot, elbow_roty, elbow)

                p.stepSimulation()
                time.sleep(dt)

            if not args.loop:
                break
    finally:
        print("Playback finished. Close the PyBullet window to exit.")
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()

