# Coordinate System Bridges

Willow Dynamics output follows the **MediaPipe Standard** (Right-Handed, Y-Down, Z-Forward). Robotics and Game Engines require different orientations.

### ROS / Isaac Sim (Z-Up)
Use this for all robotics applications where Z represents the vertical height and X represents forward depth.

**Mapping Logic:**
*   Willow Z (Depth) $\rightarrow$ ROS X (Forward)
*   Willow X (Right) $\rightarrow$ ROS -Y (Left)
*   Willow Y (Down)  $\rightarrow$ ROS -Z (Up)

```python
from willow import CoordinateBridge

# Input: (Frames, 75, 3) in MediaPipe Space
ros_seq = CoordinateBridge.to_ros_z_up(skeleton_sequence)
```

### Unity / Left-Handed (Y-Up)
Use this for AR/VR applications where Y represents the vertical height in a left-handed space.

**Mapping Logic:**
*   Willow X (Right) $\rightarrow$ Unity X (Right)
*   Willow Y (Down)  $\rightarrow$ Unity -Y (Up)
*   Willow Z (Depth) $\rightarrow$ Unity Z (Forward)

```python
from willow import CoordinateBridge

unity_seq = CoordinateBridge.to_unity_y_up(skeleton_sequence)
```