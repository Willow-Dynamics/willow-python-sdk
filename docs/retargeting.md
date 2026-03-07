# Kinematic Retargeting (IK)

The `KinematicRetargeter` converts absolute 3D point clouds into **Hierarchical Joint Rotations**. This is mandatory for Reinforcement Learning (RL), NVIDIA Isaac Sim, and Humanoid Teleoperation, which require bone rotations rather than Cartesian coordinates.

### 1. Extracting Full-Body Quaternions (New in v5.3)
The SDK constructs local coordinate frames for the primary hierarchical bones, outputting standardized Quaternions `[w, x, y, z]`.

```python
from willow import KinematicRetargeter

# Input: (Frames, 75, 3) array
quaternions = KinematicRetargeter.extract_full_body_quaternions(skeleton_sequence)

# The result is a dictionary of (Frames, 4) numpy arrays
print(quaternions["spine"])           # Output: [w, x, y, z] per frame
print(quaternions["right_upper_arm"])
print(quaternions["left_upper_leg"])
```

**Hierarchical Logic:**
The SDK uses cross-product plane mathematics to determine roll. For example, to calculate the `right_upper_arm` quaternion, the engine uses the vector from Shoulder-to-Elbow as the primary axis, and the vector from Elbow-to-Wrist to establish the hinge plane.

### 2. Extracting 1D Flexion Angles
For simpler ML models or basic physics evaluations, you can extract 1D hinge angles (in degrees).

```python
angles = KinematicRetargeter.extract_joint_angles(skeleton_sequence)

print(angles["left_elbow_flexion"]) # 1D numpy array of degrees
print(angles["right_knee_flexion"])
```