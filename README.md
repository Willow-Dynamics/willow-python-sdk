# Willow 5 Runtime SDK (Python)

[![Version](https://img.shields.io/pypi/v/willow-runtime)](https://pypi.org/project/willow-runtime/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The **Willow 5 Runtime SDK** is the local execution environment for the Willow Dynamics platform.

While the Willow Cloud is used to *train and generate* proprietary Action Models, this SDK allows you to *download and execute* those models locally on your own hardware. It acts as the bridge between Willow's biomechanical intelligence and your edge devices, robotics, or reinforcement learning simulations.

## Core Capabilities
- **Local Execution**: Run Zero-Shot Action Recognition entirely on your device using `.int8` binary signatures.
- **Sim-to-Real Retargeting**: Convert Computer Vision coordinates (MediaPipe) into ROS (Z-Up) and Unity formats natively.
- **Edge Physics**: Calculate Jerk, Acceleration, and Power without cloud latency using the ported Da Vinci engine.
- **DRM-Compliant**: Models can load securely into ephemeral RAM to protect Intellectual Property.

---

## Installation

```bash
pip install willow-runtime
```

---

## Configuration

To fetch models, you must initialize the `WillowClient` with credentials provided in your Partner Dashboard:

1.  **API URL**: The endpoint of your dedicated Willow Gateway.
2.  **API Key**: Your secure access token (authenticates the connection).
3.  **Customer ID**: Your specific Tenant ID (scopes the data access).

```python
from willow import WillowClient

client = WillowClient(
    api_url="https://api.your-gateway.com",
    api_key="sk_live_...",
    customer_id="cust_12345"
)
```

---

## Usage Guide

### 1. Active Streaming (Real-Time)
Best for **Robotics**, **Webcams**, or **Smart Gyms** where frames arrive one by one. The `step()` method manages an internal sliding window buffer for zero-latency triggering.

```python
from willow import WillowClient, WillowDetector

# 1. Load Model (Secure RAM Stream)
client = WillowClient(api_url="...", api_key="...", customer_id="...")
model = client.get_model("tactical-reload-v1")
detector = WillowDetector(model)

# 2. The Real-Time Loop
# Assume 'get_next_frame()' returns a (75, 3) numpy array of skeletal data
while True:
    current_skeleton, timestamp_ms = get_next_frame()
    
    # .step() executes in <2ms
    event = detector.step(current_skeleton, timestamp_ms)
    
    if event:
        print(f"Action Detected at {event['end_ms']}ms | Confidence: {event['confidence']:.2f}")
        # Trigger Actuator / Robot / UI here
```

### 2. Batch Analysis (Passive)
Best for **Data Science** and **Historical Video Processing**. Processes an entire sequence array at once.

```python
from willow import WillowDetector, load_local_model
import numpy as np

# Load model from disk (Offline Mode)
model = load_local_model("models/golf-swing-v4.int8")
detector = WillowDetector(model)

# Input: (Frames, 75, 3)
matches = detector.detect(full_sequence, timestamps)

print(f"Found {len(matches)} events.")
```

### 3. Robotics & Sim-to-Real
Bridges the gap between Computer Vision coordinates and Robotics standards for **NVIDIA Isaac Sim** or **ROS**.

```python
from willow import CoordinateBridge, KinematicRetargeter

# 1. Convert Coordinate Space
# MediaPipe (Y-Down) -> ROS/Isaac Sim (Z-Up)
ros_ready_sequence = CoordinateBridge.to_ros_z_up(raw_skeleton_sequence)

# 2. Extract Joint Angles for RL Training
angles = KinematicRetargeter.extract_joint_angles(ros_ready_sequence)

print(f"Elbow Flexion: {angles['right_elbow_flexion']}")
```

---

## Documentation Sections
- [Architecture & Best Practices](./docs/architecture_and_best_practices.md) - Understand how the Willow system functions.
- [Model Provisioning](./docs/provisioning.md) - How to fetch, cache, and manage secure models.
- [Coordinate Transforms](./docs/transforms.md) - Bridging Willow to ROS, Unity, and Unreal.
- [Robotics & Retargeting](./docs/retargeting.md) - Extracting joint angles for RL & Simulation.
- [Physics Evaluation](./docs/evaluation.md) - Scoring form, smoothness, and efficiency at the edge.
- [Topology Map](./docs/topology.md) - The Willow 75-point joint index reference.

## Support & Licensing
The Willow Runtime is a premium commercial service. A **Partner License** is required to provision models.
- [Request a License](https://willowdynamics.com/pages/contact)
- [Technical Support](https://willowdynamics.com/pages/contact)

&copy; 2025 Willow Dynamics. All rights reserved.