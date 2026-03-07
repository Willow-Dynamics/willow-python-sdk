# Physics Evaluation

The `PhysicsEvaluator` allows you to calculate the physical derivatives of any joint or implement (e.g., a bat or tool) in real-time.

### Path Kinematics
Calculate the 1st, 2nd, and 3rd derivatives of motion (Speed, Acceleration, Jerk).

**Requirements:**
*   Trajectory must contain at least **3 frames** of data to calculate Jerk.
*   Input coordinates should be in **Meters** for accurate physical units (m/s, m/s³).

```python
from willow import PhysicsEvaluator

# Trajectory of the right wrist: (Frames, 3)
wrist_path = skeleton_sequence[:, 16, :3] 

results = PhysicsEvaluator.calculate_derivatives(wrist_path, fps=30.0)

print(f"Peak Speed: {results['peak_speed']} m/s")
print(f"Peak Jerk (Smoothness): {results['peak_jerk']} m/s^3")
```