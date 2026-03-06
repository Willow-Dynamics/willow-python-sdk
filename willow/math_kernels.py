import numpy as np
from numba import jit
from .types import ZONES

@jit(nopython=True)
def fast_streaming_dtw_continuous(test_seq, seed_seq):
    """
    Space-Optimized O(M) Subsequence DTW. 
    Returns the normalized cost vector and match length for every frame.
    """
    N = test_seq.shape[0]
    M = seed_seq.shape[0]
    
    cost_array = np.full(N, np.inf)
    length_array = np.zeros(N, dtype=np.int32)
    if M == 0 or N == 0: 
        return cost_array, length_array
    
    D_prev = np.full(M + 1, np.inf)
    D_curr = np.full(M + 1, np.inf)
    D_prev[0] = 0.0
    
    S_prev = np.zeros(M + 1, dtype=np.int32)
    S_curr = np.zeros(M + 1, dtype=np.int32)

    for i in range(1, N + 1):
        D_curr[0] = 0.0
        S_curr[0] = i - 1 
        
        for j in range(1, M + 1):
            dist = 0.0
            for k in range(test_seq.shape[1]):
                diff = test_seq[i-1, k] - seed_seq[j-1, k]
                dist += diff * diff
            cost = np.sqrt(dist)
            
            if D_prev[j-1] <= D_prev[j] and D_prev[j-1] <= D_curr[j-1]:
                prev_cost, S_curr[j] = D_prev[j-1], S_prev[j-1]
            elif D_prev[j] <= D_curr[j-1]:
                prev_cost, S_curr[j] = D_prev[j], S_prev[j]
            else:
                prev_cost, S_curr[j] = D_curr[j-1], S_curr[j-1]
                
            D_curr[j] = cost + prev_cost
        
        # Save the normalized alignment cost ending at THIS specific frame
        cost_array[i-1] = D_curr[M] / M
        length_array[i-1] = (i - 1) - S_curr[M]

        for j in range(M + 1):
            D_prev[j] = D_curr[j]
            S_prev[j] = S_curr[j]

    return cost_array, length_array

def extract_rdm_signature(skeleton_sequence: np.ndarray, bitmask: int) -> np.ndarray:
    """
    Extracts the Scale-Invariant Relative Distance Matrix (RDM).
    Strictly adheres to the Torso-Length Normalization mandate.
    """
    # Resolve active joint indices based on binary bitmask
    active_indices =[]
    for name, (bit, indices) in ZONES.items():
        if bitmask & bit:
            active_indices.extend(indices)
            
    if not active_indices:
        active_indices = ZONES["torso"][1]  # Failsafe

    n_joints = len(active_indices)
    dim = int(n_joints * (n_joints - 1) / 2)
    frames = skeleton_sequence.shape[0]
    has_vis = skeleton_sequence.shape[2] == 4 if frames > 0 else False

    signature = np.zeros((frames, dim), dtype=np.float32)
    CONF_THRESH = 0.5 

    for f in range(frames):
        lms = skeleton_sequence[f]
        if np.all(lms == 0):
            continue
        
        # --- WILLOW 5 DIRECTIVE: Strict Torso-Length Normalization ---
        # 3D distance between the midpoint of shoulders (11, 12) and hips (23, 24)
        mid_shoulder = (lms[11][:3] + lms[12][:3]) / 2.0
        mid_hip = (lms[23][:3] + lms[24][:3]) / 2.0
        torso_length = np.linalg.norm(mid_shoulder - mid_hip)
        
        scale = max(torso_length, 0.01)  # Avoid division by zero
        
        idx = 0
        for i in range(n_joints):
            for j in range(i + 1, n_joints):
                idx1, idx2 = active_indices[i], active_indices[j]
                
                v1 = lms[idx1][3] if has_vis else 1.0
                v2 = lms[idx2][3] if has_vis else 1.0
                
                if (v1 * v2) >= (CONF_THRESH * CONF_THRESH):
                    dist = np.linalg.norm(lms[idx1][:3] - lms[idx2][:3]) / scale
                    signature[f, idx] = dist
                else:
                    signature[f, idx] = 0.0  # Mask out low confidence
                idx += 1

    return signature