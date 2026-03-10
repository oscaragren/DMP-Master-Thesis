"""
Play back a DMP‑generated left‑arm trajectory on the InMoov model in PyBullet.

Usage (from project root):

    python sim/inmoov_sim.py --trial-dir path/to/trial

where the trial directory contains `angles.npz` produced by your pipeline
(`elbow_deg` + `shoulder_deg`).

We:
- load a demo in joint space (elbow + 3 shoulder DOFs) from angles.npz
- fit a DMP and rollout a generated trajectory
- map these 4 angles to the InMoov left shoulder (3 DOFs) + left wrist roll
"""

from __future__ import annotations

import argparse
import sys
import time
import math
from pathlib import Path

import numpy as np
import pybullet as p

# Ensure project root is on sys.path for imports
_sim_dir = Path(__file__).resolve().parent
_project_root = _sim_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dmp.dmp import fit, rollout_simple
from kinematics.joint_dynamics import smooth_angles_deg
from vis.plot_dmp_trajectory import load_angles_demo


def load_dmp_trajectory(trial_dir: Path) -> tuple[np.ndarray, float]:
    """
    Fit a DMP on a demo from angles.npz in `trial_dir` and rollout a trajectory.

    Returns:
        q_rad: (T, 4) numpy array, joint angles in radians
        dt:   timestep used for the rollout (seconds, normalized time)
    """
    q_demo_deg = load_angles_demo(trial_dir)  # (T, 4), elbow + 3 shoulder DOFs
    q_demo_deg = smooth_angles_deg(q_demo_deg)

    T, n_joints = q_demo_deg.shape
    if n_joints != 4:
        raise ValueError(f"Expected 4-DOF demo, got shape {q_demo_deg.shape}")

    tau = 1.0
    dt = tau / (T - 1)

    model = fit(
        [q_demo_deg],
        tau=tau,
        dt=dt,
        n_basis_functions=15,
        alpha_canonical=4.0,
        alpha_transformation=25.0,
        beta_transformation=6.25,
    )

    q_gen_deg = rollout_simple(model, q_demo_deg[0], q_demo_deg[-1], tau=tau, dt=dt)
    q_gen_rad = np.deg2rad(q_gen_deg)
    return q_gen_rad, dt


def joint_index(body_uid: int, joint_name: str) -> int:
    """Resolve a joint name to its PyBullet index."""
    num_joints = p.getNumJoints(body_uid)
    for i in range(num_joints):
        info = p.getJointInfo(body_uid, i)
        name = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(
            info[1]
        )
        if name == joint_name:
            return i
    raise KeyError(f"Joint not found in URDF: {joint_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play back a DMP‑generated left‑arm trajectory on InMoov in PyBullet."
    )
    parser.add_argument(
        "--trial-dir",
        type=Path,
        default=_project_root
        / "test_data"
        / "processed"
        / "subject_01"
        / "lift"
        / "trial_003",
        help="Trial directory containing angles.npz (default: subject_01/reach/trial_008).",
    )
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    parser.add_argument(
        "--azimuth-offset-deg",
        type=float,
        default=-90.0,
        help="Constant offset applied to shoulder azimuth before mapping to InMoov yaw joint (deg).",
    )
    parser.add_argument(
        "--azimuth-sign",
        type=float,
        default=1.0,
        help="Sign applied to shoulder azimuth before offset (use -1 if direction is flipped).",
    )
    args = parser.parse_args()

    trial_dir: Path = args.trial_dir
    if not trial_dir.exists():
        raise FileNotFoundError(f"Trial directory not found: {trial_dir}")

    # 1. Load DMP trajectory (elbow + 3 shoulder DOFs)
    q_traj, dt = load_dmp_trajectory(trial_dir)  # (T, 4), radians
    T, n_joints = q_traj.shape

    # 2. Connect PyBullet and load InMoov URDF
    p.connect(p.GUI)
    p.setGravity(0, 0, 0)  # kinematic playback

    inmoov_dir = _sim_dir / "inmoov"
    p.setAdditionalSearchPath(str(inmoov_dir))
    urdf_path = inmoov_dir / "inmoov.urdf"
    if not urdf_path.exists():
        p.disconnect()
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # Default orientation (no extra rotation).
    base_orn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
    robot = p.loadURDF(
        str(urdf_path),
        basePosition=[0, 0, 0],
        baseOrientation=base_orn,
        useFixedBase=True,
    )

    print("Loaded InMoov joints:")
    for j in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        print(j, info[1].decode("utf-8"))

    # 3. Map DMP joints to InMoov left arm joints
    #
    # DMP order (4 DOFs), from JOINT_NAMES_4DOF:
    #   0: elbow_flexion
    #   1: shoulder_elevation
    #   2: shoulder_azimuth
    #   3: shoulder_internal_rotation
    #
    # InMoov left arm joints:
    #   l_shoulder_yaw_joint   (about Z)   <-- use shoulder_azimuth
    #   l_shoulder_pitch_joint (about Y)   <-- use shoulder_elevation
    #   l_shoulder_out_joint   (about X)   <-- use shoulder_internal_rotation
    #   l_wrist_roll_joint     (about Z)   <-- (optionally) use elbow_flexion
    l_sh_yaw = joint_index(robot, "l_shoulder_yaw_joint")
    l_sh_pitch = joint_index(robot, "l_shoulder_pitch_joint")
    l_sh_roll = joint_index(robot, "l_shoulder_out_joint")
    l_wrist = joint_index(robot, "l_wrist_roll_joint")

    # Disable motors so resetJointState fully controls pose
    num_joints = p.getNumJoints(robot)
    for j in range(num_joints):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, force=0.0)

    print(f"Starting DMP playback from {trial_dir}")
    az_offset = math.radians(float(args.azimuth_offset_deg))
    az_sign = float(args.azimuth_sign)

    try:
        while True:
            for t_idx in range(T):
                q_t = q_traj[t_idx]

                elbow = float(q_t[0])
                sh_elev = float(q_t[1])
                sh_az = float(q_t[2])
                sh_int = float(q_t[3])

                # Simple mapping as described above.
                # Align "human forward" with robot forward by applying a constant azimuth offset.
                sh_az_mapped = az_sign * sh_az + az_offset
                p.resetJointState(robot, l_sh_yaw, sh_az_mapped)
                p.resetJointState(robot, l_sh_pitch, sh_elev)
                p.resetJointState(robot, l_sh_roll, sh_int)
                p.resetJointState(robot, l_wrist, elbow)

                p.stepSimulation()
                time.sleep(dt)

            if not args.loop:
                break
    finally:
        print("Playback finished. Close the GUI window to exit.")
        p.disconnect()


if __name__ == "__main__":
    main()

