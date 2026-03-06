import time
import numpy as np
from willow import WillowDetector, load_local_model

def main():
    print("--- Willow 5: Live Edge Webcam Example ---")
    
    # 1. Load Model (Offline / Air-Gapped Mode)
    # Assumes you downloaded 'model.int8' via the Client earlier.
    print("Loading local model from disk...")
    try:
        model = load_local_model("model.int8")
    except FileNotFoundError:
        print("ERROR: 'model.int8' not found.")
        print("To run this example, please download a model first using:")
        print("  client.download_model('your-model-id', 'model.int8')")
        return

    # 2. Setup Detector
    detector = WillowDetector(model)

    print("Starting Live Camera Loop (Simulated 30 FPS)...")
    print("Press Ctrl+C to stop.")
    start_time = time.time() * 1000

    # 3. The Real-Time Loop
    try:
        # Running for 300 frames (~10 seconds)
        for i in range(300):
            # Retrieve current time in ms
            current_ts = int((time.time() * 1000) - start_time)
            
            # Capture Frame (Replace with your actual Camera/MediaPipe logic)
            # Expected Shape for a single frame: (75 joints, 3 dims)
            frame_skeleton = np.zeros((75, 3), dtype=np.float32)
            
            # 4. Step the Detector
            # This executes in <2ms on modern hardware, managing its own ring buffer internally.
            event = detector.step(frame_skeleton, current_ts)
            
            if event:
                print("\n!!! ACTION DETECTED !!!")
                print(f"Timestamp: {event['end_ms']}ms | Confidence: {event['confidence']:.2f}")
                
                # -> Trigger Actuator / Robot / UI Alert here <-

            # Simulating a 30 FPS camera delay
            time.sleep(0.033) 
            
    except KeyboardInterrupt:
        print("\nLive loop terminated by user.")
        
    print("\nLive loop finished.")

if __name__ == "__main__":
    main()