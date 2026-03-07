import time
import numpy as np
from willow import WillowClient, WillowDetector, load_local_model

def main():
    print("--- Willow 5: Live Edge Webcam Example ---")
    
    # 1. Load Model (Offline / Air-Gapped Mode)
    # For Edge Devices, we recommend downloading the model once at startup
    try:
        # Attempt to load local cache
        model = load_local_model("models/tactical-reload.int8")
        print("Loaded model from local disk.")
    except FileNotFoundError:
        print("Local model not found. Downloading from Cloud Oracle...")
        
        # Provisioning Step
        client = WillowClient(
            api_url="https://api.willowdynamics.com",
            api_key="sk_live_...",
            customer_id="cust_..."
        )
        
        # Save to disk for offline use
        client.download_model("tactical-reload-v1", "models/tactical-reload.int8")
        model = load_local_model("models/tactical-reload.int8")

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
            # It uses a 1-frame delay to confirm the peak of the action accurately.
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