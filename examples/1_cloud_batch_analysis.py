import numpy as np
from willow import WillowClient, WillowDetector, CoordinateBridge

def main():
    print("--- Willow 5: Cloud Batch Analysis Example ---")
    
    # 1. Initialize API Client
    # Replace with your actual paid enterprise API key
    client = WillowClient(api_key="YOUR_WILLOW_API_KEY")

    # 2. Download Model to RAM (Zero Disk Footprint)
    print("Fetching model into ephemeral RAM...")
    try:
        model = client.get_model(model_id="tactical-reload-v1")
    except Exception as e:
        print(f"Failed to fetch model (Check API Key): {e}")
        return

    # 3. Simulate Loading Skeletal Data 
    # (e.g., from an S3 JSON file or a MediaPipe batch output)
    # Expected Shape: (Frames, 75 joints, 3 dims)
    print("Loading batch video data...")
    dummy_skeleton = np.zeros((100, 75, 3), dtype=np.float32)
    dummy_timestamps =[t * 33 for t in range(100)]  # ~30 FPS

    # 4. Run Detection (Passive Batch Mode)
    print("Executing Continuous DTW Detection...")
    detector = WillowDetector(model)
    matches = detector.detect(dummy_skeleton, dummy_timestamps)

    print(f"\nFound {len(matches)} actions:")
    for m in matches:
        print(f" - Match from {m['start_ms']}ms to {m['end_ms']}ms (Confidence: {m['confidence']:.2f})")
        
        # 5. Export for Simulation (Sim-to-Real)
        # Extract the specific slice of data for this exact action
        action_slice = dummy_skeleton[m['start_idx']:m['end_idx']]
        
        # Convert coordinate space from MediaPipe (Y-Down) to ROS/Isaac Sim (Z-Up)
        ros_data = CoordinateBridge.to_ros_z_up(action_slice)
        print(f"   -> Retargeted {len(ros_data)} frames to ROS Z-Up standard.")

if __name__ == "__main__":
    main()