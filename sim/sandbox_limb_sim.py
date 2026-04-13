"""
Interactive sandbox for the standalone left-arm URDF in PyBullet.

This loads `sim/arm/left_arm.urdf` and exposes GUI sliders so you can manually
increase/decrease each driven joint angle.

Run from project root:

    python3 sim/sandbox_limb_sim.py
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


def joint_index(body_uid: int, joint_name: str) -> int:
    """Resolve a joint name to its PyBullet index."""
    num_joints = p.getNumJoints(body_uid)
    for i in range(num_joints):
        info = p.getJointInfo(body_uid, i)
        name = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(info[1])
        if name == joint_name:
            return i
    raise KeyError(f"Joint not found in URDF: {joint_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive joint slider sandbox for the standalone left arm.")
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0 / 240.0,
        help="Simulation step time (s).",
    )
    parser.add_argument(
        "--elbow-world-z",
        type=float,
        default=0.07,
        help="Shift the arm base so the elbow joint starts at this world Z height (m).",
    )
    args = parser.parse_args()

    p.connect(p.GUI)
    p.setGravity(0, 0, 0)  # kinematic sandbox

    p.setAdditionalSearchPath(str(_sim_dir))
    urdf_rel = "arm/new_left_arm.urdf"
    urdf_path = _sim_dir / urdf_rel
    if not urdf_path.exists():
        p.disconnect()
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # Match sim/limb_sim.py orientation (see that script + convention.md)
    base_orn = tuple(float(v) for v in p.getQuaternionFromEuler([math.pi / 2.0, 0.0, math.pi / 2.0]))
    robot = p.loadURDF(
        urdf_rel,
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=base_orn,
        useFixedBase=True,
    )

    # Driven joints
    #
    # Shoulder joints in the URDF:
    # - jLeftShoulder_rotz: abduction/adduction (axis 0 0 -1)
    # - jLeftShoulder_rotx: internal/external rotation about the upper-arm axis (axis -1 0 0)
    # - jLeftShoulder_roty: flexion/extension (axis 0 1 0)
    j_sh_rotz = joint_index(robot, "jLeftShoulder_rotz")  # shoulder abduction
    j_sh_roty = joint_index(robot, "jLeftShoulder_roty")  # shoulder flexion
    j_sh_rotx = joint_index(robot, "jLeftShoulder_rotx")  # shoulder internal rotation (highlighted ring)
    j_elbow = joint_index(robot, "jLeftElbow_roty")  # elbow flexion

    # Disable motors so resetJointState fully controls pose
    for j in range(p.getNumJoints(robot)):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, force=0.0)

    # Lift base so elbow is at desired world Z
    p.resetJointState(robot, j_elbow, 0.0)
    elbow_z0 = float(p.getLinkState(robot, j_elbow, computeForwardKinematics=True)[0][2])
    base_pos0, base_orn0 = p.getBasePositionAndOrientation(robot)
    dz = float(args.elbow_world_z) - elbow_z0
    p.resetBasePositionAndOrientation(
        robot,
        [float(base_pos0[0]), float(base_pos0[1]), float(base_pos0[2]) + dz],
        base_orn0,
    )

    # GUI sliders in degrees (more user-friendly)
    p.addUserDebugText(
        "Use the sliders to set joint angles (deg). Close the window to exit.",
        textPosition=[0.0, 0.0, 0.2],
        textColorRGB=[1.0, 1.0, 1.0],
        textSize=1.2,
        lifeTime=0.0,
    )

    s_sh_flex = p.addUserDebugParameter("shoulder_flexion (jLeftShoulder_roty) [deg]", 0.0, 80.0, 0.0)
    s_sh_abd = p.addUserDebugParameter("shoulder_abduction (jLeftShoulder_rotz) [deg]", 0.0, 40.0, 0.0)
    s_sh_int = p.addUserDebugParameter("shoulder_int_rot (jLeftShoulder_rotx) [deg]", -90.0, 90.0, 0.0)
    s_elbow = p.addUserDebugParameter("elbow_flexion (jLeftElbow_roty) [deg]", 0.0, 60.0, 0.0)

    readout_id = p.addUserDebugText(
        "",
        textPosition=[0.0, 0.0, 0.15],
        textColorRGB=[0.9, 0.9, 0.9],
        textSize=1.2,
        lifeTime=0.0,
    )

    try:
        while p.isConnected():
            sh_flex_deg = float(p.readUserDebugParameter(s_sh_flex))
            sh_abd_deg = float(p.readUserDebugParameter(s_sh_abd))
            sh_int_deg = float(p.readUserDebugParameter(s_sh_int))
            elbow_deg = float(p.readUserDebugParameter(s_elbow))

            sh_flex = math.radians(sh_flex_deg)
            # Abduction joint axis is (0,0,-1) in the URDF, so flip sign so that
            # positive slider corresponds to positive abduction convention.
            sh_abd = -math.radians(sh_abd_deg)
            sh_int = math.radians(sh_int_deg)
            elbow = math.radians(elbow_deg)

            # Shoulder joints
            p.resetJointState(robot, j_sh_rotz, sh_abd)
            p.resetJointState(robot, j_sh_roty, sh_flex)
            p.resetJointState(robot, j_sh_rotx, sh_int)
            p.resetJointState(robot, j_elbow, elbow)

            p.addUserDebugText(
                "Current angles (deg)\n"
                f"- shoulder_flexion: {sh_flex_deg:7.2f}\n"
                f"- shoulder_abduction: {sh_abd_deg:7.2f}\n"
                f"- shoulder_int_rot: {sh_int_deg:7.2f}\n"
                f"- elbow_flexion: {elbow_deg:7.2f}",
                textPosition=[0.0, 0.0, 0.15],
                textColorRGB=[0.9, 0.9, 0.9],
                textSize=1.2,
                lifeTime=0.0,
                replaceItemUniqueId=readout_id,
            )

            p.stepSimulation()
            time.sleep(float(args.dt))
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()

