import struct
import numpy as np
from willow import WillowDetector, load_local_model, parse_int8_model
from willow import CoordinateBridge, KinematicRetargeter, PhysicsEvaluator

def create_mock_int8_model():
    # Model: Version 40, Torso Bitmask (2), 5 frames
    # Scale: 1.2
    header = struct.pack('<IIffff', 40, 2, 1.2, 0.25, 3.0, 0.20)
    # Features: 127 (Int8 Max) -> Dequantized to 1.2
    mock_features = np.ones((5, 6), dtype=np.int8) * 127 
    return header + mock_features.tobytes()

def test_willow_pipeline():
    print("--- RUNNING WILLOW 5 SDK TEST SUITE ---")
    
    # 1. PARSERS
    model = parse_int8_model(create_mock_int8_model())
    assert model.config.version == 40
    print("✓ RAM-Only Parsing Passed")
    
    # 2. GENERATE DATA
    test_seq = np.zeros((20, 75, 3), dtype=np.float32)
    timestamps = [t * 33 for t in range(20)]
    
    # Base Pose (Rest)
    for f in range(20):
        test_seq[f, 11] = [-0.5, 1.5, 0.0]  # L Shoulder
        test_seq[f, 12] = [ 0.5, 1.5, 0.0]  # R Shoulder
        test_seq[f, 23] = [-0.3, 0.5, 0.0]  # L Hip (Torso Len = 1.0)
        test_seq[f, 24] = [ 0.3, 0.5, 0.0]  # R Hip
        
        # Inject Perfect Match at frames 8-12
        if 8 <= f <= 12:
            # We need Normalized Distances = 1.2
            # Torso Length = 1.0 (Shoulder Y 1.5 - Hip Y 0.5)
            # So we need Absolute Distances = 1.2
            
            # Widen Shoulders to 1.2 width (-0.6 to 0.6)
            test_seq[f, 11] = [-0.6, 1.5, 0.0]
            test_seq[f, 12] = [ 0.6, 1.5, 0.0]
            
            # Widen Hips to 1.2 width (-0.6 to 0.6)
            test_seq[f, 23] = [-0.6, 0.5, 0.0]
            test_seq[f, 24] = [ 0.6, 0.5, 0.0]
            
            # Vertical Dists (11-23) = 1.0. 
            # This will result in features [1.2, 1.2, 1.0, 1.0, 1.56, 1.56]
            # This is closer to the model [1.2, 1.2, 1.2, 1.2, 1.2, 1.2]
            # than the previous data.

    # 3. DETECTOR (PASSIVE BATCH)
    detector = WillowDetector(model)
    batch_detections = detector.detect(test_seq, timestamps)
    
    if len(batch_detections) == 0:
        print("DEBUG: No detections found. Check NMS logic or Data Scaling.")
        
    assert len(batch_detections) > 0
    print("✓ Passive Batch Detection Passed")

    # 4. DETECTOR (ACTIVE STREAMING)
    detector_live = WillowDetector(model)
    live_detections = []
    for frame, ts in zip(test_seq, timestamps):
        match = detector_live.step(frame, ts)
        if match: 
            live_detections.append(match)
            
    assert len(live_detections) > 0
    print("✓ Active Real-Time Streaming Passed")

    # 5. SPATIAL TRANSFORMS
    ros_seq = CoordinateBridge.to_ros_z_up(test_seq)
    # Check Y -> -Z translation (1.5 becomes -1.5)
    assert ros_seq[0, 11, 2] == -1.5 
    print("✓ Coordinate Bridging Passed")

    # 6. KINEMATIC RETARGETING
    angles = KinematicRetargeter.extract_joint_angles(test_seq)
    assert "left_elbow_flexion" in angles
    print("✓ Sim-to-Real IK Passed")

    # 7. PHYSICS EVALUATOR
    right_wrist_traj = test_seq[:, 16, :3]
    physics = PhysicsEvaluator.calculate_derivatives(right_wrist_traj, fps=30.0)
    assert "peak_jerk" in physics
    print("✓ Edge Physics Evaluator Passed")
    
    print("--- ALL TESTS PASSED ---")

if __name__ == "__main__":
    test_willow_pipeline()