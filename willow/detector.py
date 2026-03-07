import numpy as np
from typing import List, Dict, Optional
from .types import WillowModel
from .math_kernels import fast_streaming_dtw_continuous, extract_rdm_signature

class WillowDetector:
    """
    The Core Runtime Engine.
    Executes Zero-Shot Action Recognition using Continuous DTW.
    Supports both Batch (Passive) and Streaming (Active) modes.
    """
    
    def __init__(self, model: WillowModel):
        self.model = model
        self.cfg = model.config
        
        # --- State for Real-Time Streaming ---
        self._max_buffer_size = self.model.signature.shape[0] * 3
        self._frame_buffer = []
        self._timestamp_buffer = []
        self._last_emitted_end_ms = -1

    def detect(self, skeleton_sequence: np.ndarray, timestamps_ms: List[int], is_live: bool = False) -> List[Dict]:
        """
        Core detection loop.
        
        :param skeleton_sequence: (Frames, 75, 3) Array of 3D joints.
        :param timestamps_ms: List of timestamps corresponding to frames.
        :param is_live: Boolean flag. If True, enforces strict falling-edge logic.
        :return: List of detected event dictionaries.
        """
        # Data Validation
        if skeleton_sequence.shape[0] != len(timestamps_ms):
            raise ValueError("Sequence length must match timestamps length.")

        # 1. Feature Extraction (Raw Coordinates -> RDM Signature)
        test_sig = extract_rdm_signature(skeleton_sequence, self.cfg.zone_bitmask)
        if len(test_sig) == 0: 
            return []

        # 2. Math Kernel Execution (Numba Optimized)
        costs, lengths = fast_streaming_dtw_continuous(test_sig, self.model.signature)
        
        # 3. Similarity Normalization
        # Convert raw DTW cost (0 -> Infinity) to Similarity (1.0 -> 0.0)
        similarities = np.clip(1.0 - (costs / self.cfg.dtw_sensitivity), 0.0, 1.0)
        
        occurrences = []
        n_frames = len(similarities)
        
        # 4. Non-Maximum Suppression (NMS) - The "Peak Finder"
        for i in range(1, n_frames):
            # A. Local Maximum Check: Is this frame a better match than the previous one?
            # We use >= to allow for 'plateaus' where the score stays identical for a few frames.
            is_local_max = similarities[i] >= similarities[i-1]
            
            # B. Falling Edge Check: Has the match quality started to degrade?
            # If is_live=True, we MUST see the score drop (i > i+1) to confirm the action ended.
            # If is_live=False (Batch), the very last frame of the video is allowed to be a peak.
            if i < n_frames - 1:
                is_falling_edge = similarities[i] > similarities[i+1]
            else:
                is_falling_edge = not is_live
            
            # C. Threshold Check: Is the similarity above 0.5?
            if similarities[i] >= 0.50 and is_local_max and is_falling_edge:
                match_len = lengths[i]
                start_idx = max(0, i - match_len)
                
                occurrences.append({
                    "start_ms": timestamps_ms[start_idx],
                    "end_ms": timestamps_ms[i],
                    "confidence": float(similarities[i])
                })

        # 5. Overlap Suppression
        # If two detections overlap significantly, keep the one with higher confidence.
        occurrences.sort(key=lambda x: x['confidence'], reverse=True)
        final_occurrences = []
        
        for occ in occurrences:
            is_overlapping = False
            occ_dur = max(1, occ['end_ms'] - occ['start_ms'])
            
            for fin in final_occurrences:
                # Calculate Intersection over Union / Intersection over Duration
                overlap_dur = max(0, min(occ['end_ms'], fin['end_ms']) - max(occ['start_ms'], fin['start_ms']))
                
                if overlap_dur > 0:
                    overlap_ratio = overlap_dur / occ_dur
                    if overlap_ratio > self.cfg.overlap_tolerance:
                        is_overlapping = True
                        break
                        
            if not is_overlapping:
                final_occurrences.append(occ)

        # Sort chronologically for the user
        final_occurrences.sort(key=lambda x: x['start_ms'])
        return final_occurrences

    def step(self, current_frame_skeleton: np.ndarray, timestamp_ms: int) -> Optional[Dict]:
        """
        ACTIVE MODE: Real-Time Streaming Interface.
        
        Ingests a single frame, appends it to a rolling buffer, and checks if
        an action completed *exactly* at this timestamp.
        
        :param current_frame_skeleton: (75, 3) Array for the current instant.
        :param timestamp_ms: The current time in milliseconds.
        :return: Event Dictionary if action detected, else None.
        """
        # 1. Update Ring Buffer
        self._frame_buffer.append(current_frame_skeleton)
        self._timestamp_buffer.append(timestamp_ms)
        
        # Maintain buffer size to prevent memory leaks
        if len(self._frame_buffer) > self._max_buffer_size:
            self._frame_buffer.pop(0)
            self._timestamp_buffer.pop(0)
            
        # We need enough history to match the signature length
        if len(self._frame_buffer) < self.model.signature.shape[0]:
            return None
            
        # 2. Run Detection on the Window
        # Note: is_live=True enforces the strict falling-edge logic
        window_arr = np.array(self._frame_buffer, dtype=np.float32)
        window_matches = self.detect(window_arr, self._timestamp_buffer, is_live=True)
        
        # 3. Filter for Zero-Latency Events
        # We look for a match that ended at 'timestamp_buffer[-2]' (One frame ago).
        # Why -2? Because NMS requires a "falling edge" (next frame score is lower).
        # We can only confirm a peak once we have seen the *next* frame drop.
        target_timestamp = self._timestamp_buffer[-2] if len(self._timestamp_buffer) > 1 else -1
        
        for match in window_matches:
            # Check if this match concludes exactly at our confirmation point
            if match['end_ms'] == target_timestamp:
                # Debounce: Ensure we haven't already emitted this exact event
                if match['end_ms'] > self._last_emitted_end_ms:
                    self._last_emitted_end_ms = match['end_ms']
                    return match
                
        return None