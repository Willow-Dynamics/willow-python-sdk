# Willow 5 Runtime SDK

The official Python SDK for **Willow Dynamics**.

This SDK acts as the bridge between the Willow Cloud Oracle and your local environment. It enables **Zero-Shot Action Recognition**, **Kinematic Retargeting**, and **Physics Evaluation** on edge devices, cloud pipelines, and simulation clusters.

It is designed to be dependency-light (No AWS/Boto3 required) and privacy-first (Models run in ephemeral RAM).

---

## Installation

```bash
pip install willow-runtime
```

---

## 1. Quick Start: Model Provisioning

Securely fetch your proprietary action models from the Willow Cloud.

```python
from willow import WillowClient

# Initialize with your API Key
client = WillowClient(api_key="YOUR_WILLOW_API_KEY")

# Option A: Stream directly to RAM (Secure / Cloud / Ephemeral)
# The model never touches the physical disk.
model = client.get_model("tactical-reload-v1")

# Option B: Download to Disk (Offline / Edge / Air-Gapped)
client.download_model("tactical-reload-v1", "./models/reload.int8")
```

---

## 2. Usage: Active Streaming (Real-Time)

Best for **Robotics**, **Webcams**, or **Smart Gyms** where frames arrive one by one. The `step()` method manages an internal sliding window buffer for zero-latency triggering.

```python
from willow import WillowClient, WillowDetector

# 1. Load Model
client = WillowClient(api_key="YOUR_API_KEY")
model = client.get_model("tactical-reload-v1")
detector = WillowDetector(model)

# 2. The Real-Time Loop
# Assume 'get_next_frame()' returns a (75, 3) numpy array of skeletal data
while True:
    current_skeleton, timestamp_ms = get_next_frame()
    
    # .step() executes in <2ms on modern CPUs
    event = detector.step(current_skeleton, timestamp_ms)
    
    if event:
        print(f"!!! ACTION DETECTED !!!")
        print(f"Timestamp: {event['end_ms']}ms | Confidence: {event['confidence']:.2f}")
        
        # Trigger Actuator / Robot / UI here
```

---

## 3. Usage: Batch Analysis (Passive)

Best for **Data Science**, **Historical Video Processing**, or **Cloud ETL** pipelines.

```python
from willow import WillowClient, WillowDetector
import numpy as np

# 1. Load Model
client = WillowClient(api_key="YOUR_API_KEY")
model = client.get_model("golf-swing-v4")
detector = WillowDetector(model)

# 2. Load Data (e.g., from an uploaded video file processed by MediaPipe)
# Shape: (Frames, 75, 3)
full_sequence = np.load("recording_data.npy") 
timestamps = [t * 33 for t in range(len(full_sequence))]

# 3. Detect All Occurrences
matches = detector.detect(full_sequence, timestamps)

print(f"Found {len(matches)} events:")
for m in matches:
    print(f" - {m['start_ms']}ms to {m['end_ms']}ms (Conf: {m['confidence']:.2f})")
```

---

## 4. Usage: Sim-to-Real & Robotics

Best for **Reinforcement Learning (RL)**, **NVIDIA Isaac Sim**, and **Humanoid Teleoperation**. Bridges the gap between Computer Vision coordinates and Robotics standards.

```python
from willow import CoordinateBridge, KinematicRetargeter

# 1. Convert Coordinate Space
# MediaPipe (Y-Down) -> ROS/Isaac Sim (Z-Up)
ros_ready_sequence = CoordinateBridge.to_ros_z_up(raw_skeleton_sequence)

# 2. Extract Joint Angles for RL Training
# Returns dictionary of 1D angle arrays (e.g., "right_elbow_flexion")
joint_angles = KinematicRetargeter.extract_joint_angles(ros_ready_sequence)

print(f"Extracted {len(joint_angles)} joint features for simulation training.")
```

---
# Documentation Sections
1. [Model Provisioning](./docs/provisioning.md) - How to fetch and manage models.
2. [Coordinate Transforms](./docs/transforms.md) - Bridging Willow to ROS, Unity, and Unreal.
3. [Robotics & Retargeting](./docs/retargeting.md) - Extracting joint angles for RL & Simulation.
4. [Physics Evaluation](./docs/evaluation.md) - Scoring form and efficiency at the edge.
5. [Topology Map](./docs/topology.md) - The Willow 75-point joint index reference.

## Support & Licensing
Willow 5 Runtime is a paid service. A valid **Partner License** is required to fetch models from the Cloud Oracle. 
- [Request a License](https://willowdynamics.com)
- [Technical Support](mailto:support@willowdynamics.com)
## License

MIT License. Copyright (c) 2026 Willow Dynamics.