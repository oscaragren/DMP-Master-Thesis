import os
import sys

import math
import numpy as np

# Allow running this file directly: `python3 tests/test_get_angles.py`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kinematics.left_arm_angles import get_angles


def _ease_in_out(t01: np.ndarray) -> np.ndarray:
    t01 = np.asarray(t01, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t01, 0.0, 1.0))


def _make_reach_forward_seq(T: int = 240) -> np.ndarray:
    """
    Build a (T, N, 3) keypoint sequence where the left hand reaches forward.

    Indices used by the left-arm kinematics code:
      0: left_shoulder
      1: left_elbow
      2: left_wrist
      3: right_shoulder
      4: left_hip
      5: right_hip

    Coordinate convention used here matches the rest of the repo:
      +X = person's right
      +Y = up
      +Z = forward

    Note: `left_arm_angles.py` uses a "WORLD_UP is -Y" convention internally,
    so this synthetic sequence uses negative Y values for points "below" the shoulders.
    """
    seq = np.zeros((T, 6, 3), dtype=np.float64)

    # Static torso reference
    # Shoulders
    seq[:, 0, :] = np.array([0.0, 0.0, 0.0])   # left_shoulder
    seq[:, 3, :] = np.array([1.0, 0.0, 0.0])   # right_shoulder
    # Hips (below shoulders in -y, slightly back in -z)
    seq[:, 4, :] = np.array([0.0, -1.0, -1.0])  # left_hip
    seq[:, 5, :] = np.array([1.0, -1.0, -1.0])  # right_hip

    # Arm lengths (arbitrary but consistent)
    L1 = 0.45  # upper arm
    L2 = 0.45  # forearm

    t = np.linspace(0.0, 1.0, T)
    s = _ease_in_out(t)

    # Shoulder flexion: 0° (arm down) -> 70° forward
    sh_flex_deg = 0.0 + 70.0 * s
    sh_flex = np.deg2rad(sh_flex_deg)

    # Elbow flexion (repo convention): 175° (nearly straight) -> 110° (bent)
    el_flex_deg = 175.0 + (110.0 - 175.0) * s
    el_flex = np.deg2rad(el_flex_deg)
    internal = np.pi - el_flex  # internal angle between upper/forearm

    # Planar reach in YZ plane (x fixed). Down is negative Y, forward is +Z.
    S = seq[:, 0, :]

    upper_dir = np.stack(
        [
            np.zeros(T),
            -np.cos(sh_flex),
            np.sin(sh_flex),
        ],
        axis=1,
    )
    E = S + L1 * upper_dir

    fore_dir = np.stack(
        [
            np.zeros(T),
            -np.cos(sh_flex + internal),
            np.sin(sh_flex + internal),
        ],
        axis=1,
    )
    W = E + L2 * fore_dir

    seq[:, 1, :] = E
    seq[:, 2, :] = W

    return seq


def _play_limb_sim(q_traj_rad: np.ndarray, dt: float = 1.0 / 60.0) -> None:
    """
    Replay a (T,4) joint trajectory in PyBullet using the same mapping as `sim/limb_sim.py`.

    q_traj_rad order:
      0: elbow_flexion
      1: shoulder_flexion
      2: shoulder_abduction
      3: shoulder_internal_rotation
    """
    try:
        import pybullet as p  # type: ignore
    except Exception as e:  # pragma: no cover
        print("PyBullet not available; skipping limb playback.", repr(e))
        return

    import time

    sim_dir = os.path.join(REPO_ROOT, "sim")
    urdf_rel = os.path.join("arm", "left_arm.urdf")
    urdf_path = os.path.join(sim_dir, urdf_rel)
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    def joint_index(body_uid: int, joint_name: str) -> int:
        num_joints = p.getNumJoints(body_uid)
        for i in range(num_joints):
            info = p.getJointInfo(body_uid, i)
            name = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(info[1])
            if name == joint_name:
                return i
        raise KeyError(f"Joint not found in URDF: {joint_name}")

    p.connect(p.GUI)
    p.setGravity(0, 0, 0)
    p.setAdditionalSearchPath(sim_dir)

    base_orn = tuple(float(v) for v in p.getQuaternionFromEuler([math.pi / 2.0, 0.0, math.pi / 2.0]))
    robot = p.loadURDF(
        urdf_rel,
        basePosition=[0, 0, 0],
        baseOrientation=base_orn,
        useFixedBase=True,
    )

    sh_rotz = joint_index(robot, "jLeftShoulder_rotz")
    sh_rotx = joint_index(robot, "jLeftShoulder_rotx")
    sh_roty = joint_index(robot, "jLeftShoulder_roty")
    elbow_roty = joint_index(robot, "jLeftElbow_roty")

    # Disable motors so resetJointState fully controls pose
    for j in range(p.getNumJoints(robot)):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, force=0.0)

    try:
        for q_t in q_traj_rad:
            if not p.isConnected():
                return
            elbow = float(q_t[0])
            sh_flex = float(q_t[1])
            sh_abd = float(q_t[2])
            sh_int = float(q_t[3])

            p.resetJointState(robot, sh_rotz, sh_abd)
            p.resetJointState(robot, sh_roty, sh_flex)
            p.resetJointState(robot, sh_rotx, sh_int)
            p.resetJointState(robot, elbow_roty, elbow)

            p.stepSimulation()
            time.sleep(float(dt))
    finally:
        if p.isConnected():
            p.disconnect()


def main() -> None:
    # 1) Build a forward-reaching keypoint trajectory (shoulder/elbow/wrist + trunk points)
    seq = _make_reach_forward_seq(T=240)

    # 2) Compute angles from keypoints
    ang_deg = get_angles(seq)  # (T,4) => [el_flex, sh_flex, sh_abd, sh_rot] (deg)

    # Expected shape: (T, 4) => [elbow_flex, shoulder_flex, shoulder_abd, shoulder_rot]
    assert ang_deg.shape == (seq.shape[0], 4), f"Unexpected angles shape: {ang_deg.shape}"
    assert np.all(np.isfinite(ang_deg)), "Angles contain NaN/inf for a valid synthetic pose"

    # Sanity ranges (degrees). These are broad and meant to catch explosions, not biomechanics.
    assert np.all((ang_deg >= -360.0) & (ang_deg <= 360.0)), "Angles outside broad sanity range"

    print("seq shape:", seq.shape)
    print("angles shape:", ang_deg.shape)
    print("angles [deg] (rows=t, cols=[el_flex, sh_flex, sh_abd, sh_rot]):")
    np.set_printoptions(precision=3, suppress=True)
    print(ang_deg[:10])
    print("... (showing first 10 frames)")

    # 3) Play back in PyBullet using canonical repo order:
    #    [elbow_flexion, shoulder_flexion, shoulder_abduction, shoulder_internal_rotation]
    q_rad = np.deg2rad(ang_deg)
    _play_limb_sim(q_rad, dt=1.0 / 60.0)


if __name__ == "__main__":
    main()
