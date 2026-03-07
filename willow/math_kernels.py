import numpy as np
from numba import jit
from .types import ZONES

@jit(nopython=True)
def fast_streaming_dtw_continuous(test_seq, seed_seq):
    """
    Space-Optimized O(M) Subsequence DTW (Continuous).
    
    Calculates the alignment cost between a streaming input (test_seq)
    and a fixed template (seed_seq) for every frame index.
    
    Complexity: O(M) space, O(N*M) time.
    
    :param test_seq: Input sequence (N frames, D features)
    :param seed_seq: Template sequence (M frames, D features)
    :return: (cost_vector, match_length_vector)
    """
    N = test_seq.shape[0]
    M = seed_seq.shape[0]
    
    # Initialize output arrays (Infinity for cost, 0 for length)
    cost_array = np.full(N, np.inf)
    length_array = np.zeros(N, dtype=np.int32)
    
    if M == 0 or N == 0: 
        return cost_array, length_array
    
    # Dynamic Programming Grid Columns (Current and Previous)
    # We only store 2 columns to save memory (Space Optimization)
    D_prev = np.full(M + 1, np.inf)
    D_curr = np.full(M + 1, np.inf)
    
    # Start indices tracking to determine the duration of the match
    S_prev = np.zeros(M + 1, dtype=np.int32)
    S_curr = np.zeros(M + 1, dtype=np.int32)

    # Initialize: Cost to start at the beginning of template is 0
    D_prev[0] = 0.0

    # Iterate over Stream (Test Sequence)
    for i in range(1, N + 1):
        D_curr[0] = 0.0
        S_curr[0] = i - 1  # A new match can start at any frame i
        
        # Iterate over Template (Seed Sequence)
        for j in range(1, M + 1):
            # 1. Calculate Euclidean Distance between features
            dist = 0.0
            for k in range(test_seq.shape[1]):
                diff = test_seq[i-1, k] - seed_seq[j-1, k]
                dist += diff * diff
            cost = np.sqrt(dist)
            
            # 2. Find minimum path from previous neighbors (Insertion, Deletion, Match)
            if D_prev[j-1] <= D_prev[j] and D_prev[j-1] <= D_curr[j-1]:
                prev_cost = D_prev[j-1]
                S_curr[j] = S_prev[j-1] # Propagate start index
            elif D_prev[j] <= D_curr[j-1]:
                prev_cost = D_prev[j]
                S_curr[j] = S_prev[j]
            else:
                prev_cost = D_curr[j-1]
                S_curr[j] = S_curr[j-1]
                
            D_curr[j] = cost + prev_cost
        
        # 3. Store the Normalized Cost for a full match ending at this frame
        # We normalize by M (length of template) to make score independent of duration
        cost_array[i-1] = D_curr[M] / M
        length_array[i-1] = (i - 1) - S_curr[M]

        # Swap columns for next iteration
        for j in range(M + 1):
            D_prev[j] = D_curr[j]
            S_prev[j] = S_curr[j]

    return cost_array, length_array

def extract_rdm_signature(skeleton_sequence: np.ndarray, bitmask: int) -> np.ndarray:
    """
    Extracts the Scale-Invariant Relative Distance Matrix (RDM).
    Strictly adheres to the Torso-Length Normalization mandate.
    """
    # Safety: Ensure float32 to prevent Numba/C-API type mismatch errors
    if skeleton_sequence.dtype != np.float32:
        skeleton_sequence = skeleton_sequence.astype(np.float32)

    # 1. Resolve Active Indices from Bitmask
    active_indices = []
    for name, (bit, indices) in ZONES.items():
        if bitmask & bit:
            active_indices.extend(indices)
            
    # Failsafe: Default to Torso if bitmask is invalid
    if not active_indices:
        active_indices = ZONES["torso"][1]

    # 2. Setup Output Array
    n_joints = len(active_indices)
    dim = int(n_joints * (n_joints - 1) / 2) # Combinatorial dimension nC2
    frames = skeleton_sequence.shape[0]
    
    signature = np.zeros((frames, dim), dtype=np.float32)
    
    if frames == 0:
        return signature

    # Visibility Check: Does the input have a 4th channel (Confidence)?
    has_vis = (skeleton_sequence.shape[-1] == 4)
    CONF_THRESH = 0.5 

    # 3. Extraction Loop
    for f in range(frames):
        lms = skeleton_sequence[f]
        
        # Torso Normalization Logic
        mid_shoulder = (lms[11, :3] + lms[12, :3]) / 2.0
        mid_hip = (lms[23, :3] + lms[24, :3]) / 2.0
        torso_length = np.linalg.norm(mid_shoulder - mid_hip)
        
        # Safety clamp to prevent division by zero
        scale = max(torso_length, 0.01)
        
        idx = 0
        # Compute pairwise distances for all active joints
        for i in range(n_joints):
            for j in range(i + 1, n_joints):
                idx1 = active_indices[i]
                idx2 = active_indices[j]
                
                # Check visibility confidence if available
                v1 = lms[idx1, 3] if has_vis else 1.0
                v2 = lms[idx2, 3] if has_vis else 1.0
                
                if (v1 * v2) >= (CONF_THRESH * CONF_THRESH):
                    # Euclidean Distance / Torso Scale
                    dist = np.linalg.norm(lms[idx1, :3] - lms[idx2, :3]) / scale
                    signature[f, idx] = dist
                else:
                    signature[f, idx] = 0.0 # Mask out low-confidence data
                
                idx += 1

    return signature