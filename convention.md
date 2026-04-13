## Definition
RS = Right shoulder, LS = Left shoulder
- X = norm(LS - RS)
- Y = norm(Shoulder_center - Hip_center)
- Z = cross(X, Y)

Re-orthogonalize:
- Y = cross(Z, X)

This means that:
- **Trunk +X** points from **right shoulder → left shoulder**
- **Trunk +Y** points from **hip center → shoulder center**
- **Trunk +Z** points roughly **forward** (out of chest), from `cross(X, Y)`

Notes:
- In `kinematics/simple_kinematics.py` the input 3D keypoints are in the **OAK-D camera frame** (X right, Y down, Z forward). The trunk frame above is constructed in that camera frame and then used as a rotating reference frame.
- When computing shoulder flexion/abduction we treat “arm down” as the direction **-Y in trunk frame**, so the formulas use `-upper_arm_trunk_y`.

## Definition of angle conventions

### Shoulder flexion
Flexion is positive when the arm moves toward **trunk +Z**.
So:
- **Arm forward (+Z)** -> positive (≈ +90°)
- **Arm backward (−Z)** -> negative
- **Arm down (−Y)** = 0°

### Shoulder abduction
Abduction is positive when the arm moves toward **trunk +X**.
So:
- **Arm toward +X** -> positive (≈ +90°)
- **Arm toward −X** -> negative

### Shoulder internal rotation (lateral/medial rotation proxy)
Rotation around the upper-arm (humerus) axis, estimated from the **forearm direction** after undoing flexion/abduction and projecting onto the trunk XZ plane.

Convention (intended for poses with elbow bent ~90°):
- **0°**: forearm parallel to **trunk +Z**
- **−90°**: forearm parallel to **trunk +X**
- **+90°**: forearm parallel to **trunk −X**

### Elbow flexion
Elbow flexion is defined as the **angle between upper arm and forearm**:
- **0°**: fully extended (segments aligned)
- **90°**: elbow bent to a right angle



