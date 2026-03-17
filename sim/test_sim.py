"""
Spawn the standalone left arm (limb) in PyBullet and hold it fixed.

This script mirrors the arm-loading "starting pose" from `sim/limb_sim.py`
(same URDF + base orientation), but does **not** play any trajectory.
Instead it keeps the arm fixed at its starting joint configuration.
"""
import time
import math
from pathlib import Path

import pybullet as p


def get_project_root() -> Path:
    """Return project root (parent of sim/)."""
    sim_dir = Path(__file__).resolve().parent
    return sim_dir.parent


def main() -> None:
    project_root = get_project_root()
    sim_dir = Path(__file__).resolve().parent

    # 1) Start PyBullet GUI and load the standalone arm URDF
    p.connect(p.GUI)
    p.setGravity(0, 0, 0)

    p.setAdditionalSearchPath(str(sim_dir))
    urdf_rel = "arm/left_arm.urdf"
    urdf_path = sim_dir / urdf_rel
    if not urdf_path.exists():
        p.disconnect()
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # Match limb_sim's base orientation so zero joints look like the "starting pose".
    base_orn = p.getQuaternionFromEuler([-math.pi / 2.0, 0.0, 0.0])
    robot = p.loadURDF(
        urdf_rel,
        basePosition=[0, 0, 0],
        baseOrientation=base_orn,
        useFixedBase=True,
    )

    # 2) Capture the starting joint positions (the current URDF defaults).
    num_joints = p.getNumJoints(robot)
    q0 = [float(p.getJointState(robot, j)[0]) for j in range(num_joints)]

    # 3) Hold the starting pose with position control (even if gravity is later enabled).
    for j in range(num_joints):
        p.setJointMotorControl2(
            robot,
            j,
            p.POSITION_CONTROL,
            targetPosition=q0[j],
            force=50.0,
        )

    print("Standalone left arm loaded. Holding it fixed at the starting pose. Close the GUI window to exit.")
    try:
        while True:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()