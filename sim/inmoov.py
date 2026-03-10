import pybullet as p
import pybullet_data
from pathlib import Path

import time
import math

p.connect(p.GUI)
p.setGravity(0,0,-9.81)

sim_dir = Path(__file__).resolve().parent
inmoov_dir = sim_dir / "inmoov"
# Ensure Bullet can resolve `meshes/...` referenced by the URDF.
p.setAdditionalSearchPath(str(inmoov_dir))
urdf_path = inmoov_dir / "inmoov.urdf"
if not urdf_path.exists():
    p.disconnect()
    raise FileNotFoundError(f"URDF not found: {urdf_path}")

robot = p.loadURDF(
    str(urdf_path),
    basePosition=[0,0,0],
    useFixedBase=True
)

def joint_index(body_uid: int, joint_name: str) -> int:
    for i in range(p.getNumJoints(body_uid)):
        info = p.getJointInfo(body_uid, i)
        name = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(info[1])
        if name == joint_name:
            return i
    raise KeyError(f"Joint not found: {joint_name}")


print("Loaded InMoov. Joints:")
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    print(i, info[1].decode("utf-8"))

# Simple left-arm motion (shoulder ab/adduct + wrist roll) so you can see movement.
# Uses kinematic reset (works even without accurate inertials).
l_shoulder_yaw = joint_index(robot, "l_shoulder_yaw_joint")
l_shoulder_pitch = joint_index(robot, "l_shoulder_pitch_joint")
l_shoulder_roll = joint_index(robot, "l_shoulder_out_joint")
l_wrist = joint_index(robot, "l_wrist_roll_joint")

dt = 1 / 240
t0 = time.time()

while p.isConnected():
    t = time.time() - t0

    # Move 3 shoulder axes sequentially: yaw (Z) -> pitch (Y) -> roll (X), repeating.
    # Only one axis moves at a time, the others are held at 0.
    phase_s = 4.0
    phase = int(t // phase_s) % 3
    tau = t - (t // phase_s) * phase_s  # time within phase
    amp = math.radians(35.0)
    target = amp * math.sin(2.0 * math.pi * (1.0 / phase_s) * tau)

    yaw_t = target if phase == 0 else 0.0
    pitch_t = target if phase == 1 else 0.0
    roll_t = target if phase == 2 else 0.0

    p.resetJointState(robot, l_shoulder_yaw, yaw_t)
    p.resetJointState(robot, l_shoulder_pitch, pitch_t)
    p.resetJointState(robot, l_shoulder_roll, roll_t)

    # Keep a small wrist motion so the hand is easier to spot.
    wrist_target = math.radians(25.0) * math.sin(2.0 * math.pi * 0.5 * t)
    p.resetJointState(robot, l_wrist, wrist_target)

    p.stepSimulation()
    time.sleep(dt)