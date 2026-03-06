# Coordinate System Bridges

Willow Dynamics output follows the **MediaPipe Standard** (Right-Handed, Y-Down, Z-Forward). Robotics and Game Engines require different orientations.

### ROS / Isaac Sim (Z-Up)
Use this for all robotics applications where Z represents the vertical height.

```python
from willow import CoordinateBridge

# Translates MP_X -> ROS_X, MP_Z -> ROS_Y, -MP_Y -> ROS_Z
ros_seq = CoordinateBridge.to_ros_z_up(skeleton_sequence)
```

### Unity / Left-Handed (Y-Up)
Use this for AR/VR applications where Y represents the vertical height in a left-handed space.

```python
from willow import CoordinateBridge

# Translates MP_X -> Unity_X, -MP_Y -> Unity_Y, MP_Z -> Unity_Z
unity_seq = CoordinateBridge.to_unity_y_up(skeleton_sequence)
```