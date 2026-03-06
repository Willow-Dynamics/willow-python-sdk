import numpy as np
from typing import List, Dict, Optional
from .types import WillowModel
from .math_kernels import fast_streaming_dtw_continuous, extract_rdm_signature

class WillowDetector:
    """
    Core Runtime Engine for executing Zero-Shot Action Recognition locally.
    """
    def __init__(self, model: WillowModel):
        self.model = model
        self.cfg = model.config
        
        # State architecture for the Real-Time Streaming functionality
        self._max_buffer_size = self.model.signature.shape[0] * 3
        self._frame_buffer =[]
        self._timestamp_buffer =[]
        self._last_emitted_end_ms = -1

    def detect(self, skeleton_sequence: np.ndarray, timestamps_ms: List[int]) -> List[Dict]:
        """
        PASSIVE MODE: Batch execution. Processes a full video array at once.
        skeleton_sequence: Expected shape (Frames, 75, 3) or (Frames, 75, 4).
        """
        if skeleton_sequence.shape[0] != len(timestamps_ms):
            raise ValueError("Skeleton sequence length must match timestamps array length.")

        # 1. RDM Feature Extraction (Torso-Normalized)
        test_sig = extract_rdm_signature(skeleton_sequence, self.cfg.zone_bitmask)
        if len(test_sig) == 0: 
            return[]

        # 2. Continuous DTW Kernel
        costs, lengths = fast_streaming_dtw_continuous(test_sig, self.model.signature)
        
        # Normalize costs to similarities (0.0 to 1.0)
        similarities = np.clip(1.0 - (costs / self.cfg.dtw_sensitivity), 0.0, 1.0)
        
        occurrences =[]
        n_frames = len(similarities)
        
        # 3. NMS with Terminal Peak Logic (Robust to Plateaus)
        for i in range(1, n_frames):
            # Check if this frame is a local maximum (Allowing plateaus via >=)
            is_local_max = similarities[i] >= similarities[i-1]
            
            # Check falling edge (Strictly greater than next, or end of sequence)
            # This ensures we pick the *end* of a plateau as the event timestamp
            is_falling_edge = (i == n_frames - 1) or (similarities[i] > similarities[i+1])
            
            if similarities[i] >= 0.50 and is_local_max and is_falling_edge:
                match_len = lengths[i]
                start_idx = max(0, i - match_len)
                
                occurrences.append({
                    "start_ms": timestamps_ms[start_idx],
                    "end_ms": timestamps_ms[i],
                    "confidence": float(similarities[i]),
                    "start_idx": start_idx,
                    "end_idx": i
                })

        # 4. Temporal Overlap Suppression (NMS)
        occurrences.sort(key=lambda x: x['confidence'], reverse=True)
        final_occurrences =[]
        
        for occ in occurrences:
            is_overlapping = False
            occ_dur = max(1, occ['end_ms'] - occ['start_ms'])
            
            for fin in final_occurrences:
                latest_start = max(occ['start_ms'], fin['start_ms'])
                earliest_end = min(occ['end_ms'], fin['end_ms'])
                overlap_dur = max(0, earliest_end - latest_start)
                
                if overlap_dur > 0:
                    overlap_ratio = overlap_dur / occ_dur
                    if overlap_ratio > self.cfg.overlap_tolerance:
                        is_overlapping = True
                        break
                        
            if not is_overlapping:
                final_occurrences.append(occ)

        # Return chronologically sorted
        final_occurrences.sort(key=lambda x: x['start_ms'])
        
        # Clean internal indices for pristine API output
        for o in final_occurrences:
            del o['start_idx']
            del o['end_idx']
            
        return final_occurrences

    def step(self, current_frame_skeleton: np.ndarray, timestamp_ms: int) -> Optional[Dict]:
        """
        ACTIVE MODE: Real-Time Streaming loop for live webcam or robotics edge processing.
        Ingests a single frame (75, 3) and yields an event the millisecond an action completes.
        """
        # 1. Manage Ring Buffer
        self._frame_buffer.append(current_frame_skeleton)
        self._timestamp_buffer.append(timestamp_ms)
        
        if len(self._frame_buffer) > self._max_buffer_size:
            self._frame_buffer.pop(0)
            self._timestamp_buffer.pop(0)
            
        # Needs minimum history to run DTW
        if len(self._frame_buffer) < self.model.signature.shape[0]:
            return None
            
        # 2. Execute Batch Detect on the sliding window
        window_arr = np.array(self._frame_buffer)
        window_matches = self.detect(window_arr, self._timestamp_buffer)
        
        # 3. Filter for Zero-Latency Events
        for match in window_matches:
            # If the match completed EXACTLY on this millisecond, emit it.
            # And ensure we haven't already emitted it due to sliding window overlap.
            if match['end_ms'] == timestamp_ms and match['end_ms'] > self._last_emitted_end_ms:
                self._last_emitted_end_ms = match['end_ms']
                return match
                
        return None