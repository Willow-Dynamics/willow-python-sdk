# Willow 75-Point Topology Map

All skeletal inputs to the `detect()` or `step()` methods must adhere to this index map.

### Humanoid Joint Indices
| Index | Joint Name | Notes |
| :--- | :--- | :--- |
| **0** | Nose | |
| **11, 12** | Shoulders | Used for Torso Normalization |
| **13, 14** | Elbows | |
| **15, 16** | Wrists | |
| **23, 24** | Hips | Used for Torso Normalization |
| **25, 26** | Knees | |
| **27, 28** | Ankles | |
| **31, 32** | Toes | |

### Hand Indices (Hi-Fi)
| Range | Target |
| :--- | :--- |
| **33 - 53** | Left Hand | 21 joints (MCP, PIP, DIP, TIP) |
| **54 - 74** | Right Hand | 21 joints (MCP, PIP, DIP, TIP) |

### Normalization
Willow signatures are **Torso-Length Scaled**. We calculate the 3D distance between the Shoulder Midpoint and Hip Midpoint to ensure the "size" of the human does not affect recognition accuracy.