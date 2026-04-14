import math
from pathlib import Path
import time

import pybullet as p


"""
Spawn the standalone left arm (limb) in PyBullet and sweep elbow flexion.

This is a small debugging script for joint axes and basic motion.
"""


def joint_index(body_uid: int, joint_name: str) -> int:
    """Resolve a joint name to its PyBullet index."""
    num_joints = p.getNumJoints(body_uid)
    for i in range(num_joints):
        info = p.getJointInfo(body_uid, i)
        name = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(info[1])
        if name == joint_name:
            return i
    raise KeyError(f"Joint not found in URDF: {joint_name}")


def _quat_mul(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _rotate_vec_by_quat(v: tuple[float, float, float], q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    # v' = q * (v,0) * q_conj
    x, y, z, w = q
    qv = (v[0], v[1], v[2], 0.0)
    q_conj = (-x, -y, -z, w)
    t = _quat_mul(q, qv)
    r = _quat_mul(t, q_conj)
    return (r[0], r[1], r[2])


def _world_joint_axis(robot: int, joint_idx: int) -> tuple[float, float, float]:
    """Return joint axis expressed in world coordinates."""
    info = p.getJointInfo(robot, joint_idx)
    joint_axis = tuple(float(a) for a in info[13])
    parent_idx = int(info[16])
    parent_frame_orn = tuple(float(a) for a in info[15])

    if parent_idx == -1:
        parent_world_orn = tuple(float(a) for a in p.getBasePositionAndOrientation(robot)[1])
    else:
        parent_world_orn = tuple(float(a) for a in p.getLinkState(robot, parent_idx, computeForwardKinematics=True)[1])

    joint_world_orn = _quat_mul(parent_world_orn, parent_frame_orn)
    return _rotate_vec_by_quat(joint_axis, joint_world_orn)


def _quat_from_axis_angle(axis: tuple[float, float, float], angle_rad: float) -> tuple[float, float, float, float]:
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    ax /= n
    ay /= n
    az /= n
    s = math.sin(0.5 * angle_rad)
    c = math.cos(0.5 * angle_rad)
    return (ax * s, ay * s, az * s, c)

def _quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = q
    return (-x, -y, -z, w)

def _arm_base_orientation() -> tuple[float, float, float, float]:
    # 1) "Hang" rotation so the arm looks anatomical at zero joint angles.
    q_hang = tuple(float(v) for v in p.getQuaternionFromEuler([math.pi/2.0, 0.0, math.pi/2.0]))

    return q_hang


def main() -> None:
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

    base_orn = tuple(float(v) for v in p.getQuaternionFromEuler([math.pi/2.0, 0.0, math.pi/2.0]))
    robot = p.loadURDF(
        urdf_rel,
        basePosition=[0, 0, 0],
        baseOrientation=base_orn,
        useFixedBase=True,
    )

    # 2) Capture the starting joint positions (the current URDF defaults).
    num_joints = p.getNumJoints(robot)
    q0 = [float(p.getJointState(robot, j)[0]) for j in range(num_joints)]

    # 3) Drive elbow flexion from 90 deg to 40 deg (and back), repeatedly.
    elbow_roty = joint_index(robot, "jLeftElbow_roty")
    elbow_hi = math.radians(90.0)
    elbow_lo = math.radians(40.0)

    # 4) Hold the starting pose with position control for all joints.
    for j in range(num_joints):
        p.setJointMotorControl2(
            robot,
            j,
            p.POSITION_CONTROL,
            targetPosition=q0[j],
            force=50.0,
        )

    # Quick confirmation: print shoulder joint axes in URDF frame and in world frame.
    sh_rotz = joint_index(robot, "jLeftShoulder_rotz")
    sh_rotx = joint_index(robot, "jLeftShoulder_rotx")
    sh_roty = joint_index(robot, "jLeftShoulder_roty")
    print("Shoulder joint axis check (URDF axis -> world axis):")
    for name, idx in [
        ("jLeftShoulder_rotz", sh_rotz),
        ("jLeftShoulder_rotx", sh_rotx),
        ("jLeftShoulder_roty", sh_roty),
    ]:
        info = p.getJointInfo(robot, idx)
        urdf_axis = tuple(float(a) for a in info[13])
        world_axis = _world_joint_axis(robot, idx)
        print(f"  {name}: {urdf_axis} -> {tuple(round(a, 3) for a in world_axis)}")

    print(
        f"Standalone left arm loaded. Sweeping elbow flexion ({elbow_hi:.3f} -> {elbow_lo:.3f} rad). "
        "Close the GUI window to exit."
    )
    try:
        t0 = time.time()
        while True:
            # If the GUI window is closed, the physics server disconnects.
            # Exit cleanly instead of raising "Not connected to physics server".
            if not p.isConnected():
                break
            t = time.time() - t0
            # Smooth periodic sweep: maps sin() to [elbow_lo, elbow_hi].
            s = 0.5 * (1.0 + math.sin(2.0 * math.pi * 0.15 * t))  # 0..1
            target = elbow_hi + (elbow_lo - elbow_hi) * s
            p.setJointMotorControl2(
                robot,
                elbow_roty,
                p.POSITION_CONTROL,
                targetPosition=float(target),
                force=50.0,
            )
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()