import struct
import numpy as np
from willow import WillowDetector, load_local_model, parse_int8_model
from willow import CoordinateBridge, KinematicRetargeter, PhysicsEvaluator

def create_mock_int8_model():
    header = struct.pack('<IIffff', 40, 2, 1.2, 0.25, 3.0, 0.20)
    mock_features = np.ones((5, 6), dtype=np.int8) * 127 
    return header + mock_features.tobytes()

def test_willow_pipeline():
    print("--- RUNNING WILLOW 5 SDK TEST SUITE (v5.3.0) ---")
    
    model = parse_int8_model(create_mock_int8_model())
    assert model.config.version == 40
    print("✓ RAM-Only Parsing Passed")
    
    test_seq = np.zeros((20, 75, 3), dtype=np.float32)
    timestamps = [t * 33 for t in range(20)]
    for f in range(20):
        # Build a valid skeleton to prevent zero-vector math errors
        test_seq[f, 11] = [-0.5, 1.5, 0.0]  # L Shoulder
        test_seq[f, 12] =[ 0.5, 1.5, 0.0]  # R Shoulder
        test_seq[f, 13] =[-0.5, 1.0, 0.0]  # L Elbow
        test_seq[f, 14] = [ 0.5, 1.0, 0.0]  # R Elbow
        test_seq[f, 15] =[-0.5, 0.5, 0.0]  # L Wrist
        test_seq[f, 16] = [ 0.5, 0.5, 0.0]  # R Wrist
        test_seq[f, 23] = [-0.3, 0.5, 0.0]  # L Hip
        test_seq[f, 24] =[ 0.3, 0.5, 0.0]  # R Hip
        test_seq[f, 25] = [-0.3, 0.0, 0.0]  # L Knee
        test_seq[f, 26] =[ 0.3, 0.0, 0.0]  # R Knee
        test_seq[f, 27] =[-0.3,-0.5, 0.0]  # L Ankle
        test_seq[f, 28] = [ 0.3,-0.5, 0.0]  # R Ankle
        
        if 8 <= f <= 12:
            test_seq[f, 11] = [-0.6, 1.5, 0.0]
            test_seq[f, 12] = [ 0.6, 1.5, 0.0]
            test_seq[f, 23] =[-0.6, 0.5, 0.0] 
            test_seq[f, 24] =[ 0.6, 0.5, 0.0]

    detector = WillowDetector(model)
    assert len(detector.detect(test_seq, timestamps)) > 0
    print("✓ Passive Batch Detection Passed")

    # TEST: KINEMATIC RETARGETING (NEW v5.3.0 QUATERNIONS)
    quats = KinematicRetargeter.extract_full_body_quaternions(test_seq)
    assert "spine" in quats
    assert quats["spine"].shape == (20, 4) # 20 frames, 4 quat dimensions[w, x, y, z]
    assert np.allclose(np.linalg.norm(quats["spine"], axis=1), 1.0) # Quats must be normalized
    print("✓ Sim-to-Real IK (Quaternions) Passed")

    angles = KinematicRetargeter.extract_joint_angles(test_seq)
    assert "left_elbow_flexion" in angles
    print("✓ Legacy 1D Flexion IK Passed")
    
    print("--- ALL TESTS PASSED ---")

if __name__ == "__main__":
    test_willow_pipeline()