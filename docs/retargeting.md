# Kinematic Retargeting

For Reinforcement Learning (RL) and Sim-to-Real applications, joint angles (1D/3D) are more useful than absolute 3D coordinates.

### Extracting 1D Joint Angles
The `KinematicRetargeter` calculates Euler-approximated flexion angles for critical biomechanical chain links.

```python
from willow import KinematicRetargeter

# Input: (Frames, 75, 3) array
angles = KinematicRetargeter.extract_joint_angles(skeleton_sequence)

# The result is a dictionary of numpy arrays
print(angles["left_elbow_flexion"]) 
print(angles["right_knee_flexion"])
```

### Target Joints
The SDK currently provides flexion/extension angles for:
- `left_elbow_flexion`
- `right_elbow_flexion`
- `left_knee_flexion`
- `right_knee_flexion`