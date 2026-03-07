import numpy as np

class KinematicRetargeter:
    """
    Translates absolute 3D point clouds into Hierarchical Joint Rotations.
    Outputs standard Quaternions[w, x, y, z] for Reinforcement Learning
    and Sim-to-Real robotics pipelines (ROS, Isaac Sim, URDF formats).
    """

    @staticmethod
    def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Fallback for 1D flexion calculations."""
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        mask = (n1 > 1e-6) & (n2 > 1e-6)
        angles = np.zeros(v1.shape[0], dtype=np.float32)
        if np.any(mask):
            dot_prod = np.sum(v1[mask] * v2[mask], axis=1)
            cos_theta = np.clip(dot_prod / (n1[mask] * n2[mask]), -1.0, 1.0)
            angles[mask] = np.degrees(np.arccos(cos_theta))
        return angles

    @staticmethod
    def _construct_basis(v_forward: np.ndarray, v_reference: np.ndarray) -> np.ndarray:
        """
        Constructs a 3x3 orthonormal rotation matrix from two vectors.
        v_forward: The primary axis (the bone itself).
        v_reference: A vector in the hinge plane (to determine roll).
        """
        v_forward = v_forward / (np.linalg.norm(v_forward, axis=-1, keepdims=True) + 1e-8)
        v_reference = v_reference / (np.linalg.norm(v_reference, axis=-1, keepdims=True) + 1e-8)
        
        v_right = np.cross(v_reference, v_forward)
        v_right = v_right / (np.linalg.norm(v_right, axis=-1, keepdims=True) + 1e-8)
        
        v_up = np.cross(v_forward, v_right)
        v_up = v_up / (np.linalg.norm(v_up, axis=-1, keepdims=True) + 1e-8)
        
        return np.stack([v_right, v_up, v_forward], axis=-1)

    @staticmethod
    def _matrix_to_quaternion(m: np.ndarray) -> np.ndarray:
        """
        Vectorized conversion of (N, 3, 3) Rotation Matrices to (N, 4) Quaternions [w, x, y, z].
        Uses pure NumPy for dependency-free edge execution.
        """
        N = m.shape[0]
        q = np.zeros((N, 4), dtype=np.float32)
        tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
        
        # Path 1: Trace > 0 (Standard conversion)
        mask0 = tr > 0
        if np.any(mask0):
            S0 = np.sqrt(tr[mask0] + 1.0) * 2.0
            q[mask0, 0] = 0.25 * S0
            q[mask0, 1] = (m[mask0, 2, 1] - m[mask0, 1, 2]) / S0
            q[mask0, 2] = (m[mask0, 0, 2] - m[mask0, 2, 0]) / S0
            q[mask0, 3] = (m[mask0, 1, 0] - m[mask0, 0, 1]) / S0

        # Path 2: Trace <= 0 (Singularity safe fallbacks)
        mask_else = ~mask0
        if np.any(mask_else):
            m_else = m[mask_else]
            diag = np.stack([m_else[:, 0, 0], m_else[:, 1, 1], m_else[:, 2, 2]], axis=-1)
            max_d = np.argmax(diag, axis=-1)
            
            for idx, c_idx in enumerate(max_d):
                real_idx = np.where(mask_else)[0][idx]
                if c_idx == 0:
                    S = np.sqrt(1.0 + m[real_idx, 0, 0] - m[real_idx, 1, 1] - m[real_idx, 2, 2]) * 2.0
                    q[real_idx, 0] = (m[real_idx, 2, 1] - m[real_idx, 1, 2]) / S
                    q[real_idx, 1] = 0.25 * S
                    q[real_idx, 2] = (m[real_idx, 0, 1] + m[real_idx, 1, 0]) / S
                    q[real_idx, 3] = (m[real_idx, 0, 2] + m[real_idx, 2, 0]) / S
                elif c_idx == 1:
                    S = np.sqrt(1.0 + m[real_idx, 1, 1] - m[real_idx, 0, 0] - m[real_idx, 2, 2]) * 2.0
                    q[real_idx, 0] = (m[real_idx, 0, 2] - m[real_idx, 2, 0]) / S
                    q[real_idx, 1] = (m[real_idx, 0, 1] + m[real_idx, 1, 0]) / S
                    q[real_idx, 2] = 0.25 * S
                    q[real_idx, 3] = (m[real_idx, 1, 2] + m[real_idx, 2, 1]) / S
                else:
                    S = np.sqrt(1.0 + m[real_idx, 2, 2] - m[real_idx, 0, 0] - m[real_idx, 1, 1]) * 2.0
                    q[real_idx, 0] = (m[real_idx, 1, 0] - m[real_idx, 0, 1]) / S
                    q[real_idx, 1] = (m[real_idx, 0, 2] + m[real_idx, 2, 0]) / S
                    q[real_idx, 2] = (m[real_idx, 1, 2] + m[real_idx, 2, 1]) / S
                    q[real_idx, 3] = 0.25 * S
        
        # Normalize Quaternions
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        return q / (norms + 1e-8)

    @staticmethod
    def extract_full_body_quaternions(skeleton_seq: np.ndarray) -> dict:
        """
        Calculates Global Quaternions [w, x, y, z] for the primary hierarchical bones.
        These can be multiplied by the inverse of their parent to yield Local Rotations.
        """
        if skeleton_seq.shape[0] == 0:
            return {}

        L_SH, R_SH = 11, 12
        L_EL, R_EL = 13, 14
        L_WR, R_WR = 15, 16
        L_HI, R_HI = 23, 24
        L_KN, R_KN = 25, 26
        L_AN, R_AN = 27, 28

        quats = {}

        # 1. Torso/Spine (Reference: Shoulders and Hips)
        mid_shoulder = (skeleton_seq[:, L_SH, :3] + skeleton_seq[:, R_SH, :3]) / 2.0
        mid_hip = (skeleton_seq[:, L_HI, :3] + skeleton_seq[:, R_HI, :3]) / 2.0
        spine_fwd = mid_shoulder - mid_hip
        spine_ref = skeleton_seq[:, L_SH, :3] - skeleton_seq[:, R_SH, :3] # Across shoulders
        spine_mat = KinematicRetargeter._construct_basis(spine_fwd, spine_ref)
        quats["spine"] = KinematicRetargeter._matrix_to_quaternion(spine_mat)

        # 2. Left Arm
        l_arm_fwd = skeleton_seq[:, L_EL, :3] - skeleton_seq[:, L_SH, :3]
        l_arm_ref = skeleton_seq[:, L_WR, :3] - skeleton_seq[:, L_EL, :3] # Plane of the elbow
        l_arm_mat = KinematicRetargeter._construct_basis(l_arm_fwd, l_arm_ref)
        quats["left_upper_arm"] = KinematicRetargeter._matrix_to_quaternion(l_arm_mat)

        # 3. Right Arm
        r_arm_fwd = skeleton_seq[:, R_EL, :3] - skeleton_seq[:, R_SH, :3]
        r_arm_ref = skeleton_seq[:, R_WR, :3] - skeleton_seq[:, R_EL, :3]
        r_arm_mat = KinematicRetargeter._construct_basis(r_arm_fwd, r_arm_ref)
        quats["right_upper_arm"] = KinematicRetargeter._matrix_to_quaternion(r_arm_mat)

        # 4. Left Leg
        l_leg_fwd = skeleton_seq[:, L_KN, :3] - skeleton_seq[:, L_HI, :3]
        l_leg_ref = skeleton_seq[:, L_AN, :3] - skeleton_seq[:, L_KN, :3]
        l_leg_mat = KinematicRetargeter._construct_basis(l_leg_fwd, l_leg_ref)
        quats["left_upper_leg"] = KinematicRetargeter._matrix_to_quaternion(l_leg_mat)

        # 5. Right Leg
        r_leg_fwd = skeleton_seq[:, R_KN, :3] - skeleton_seq[:, R_HI, :3]
        r_leg_ref = skeleton_seq[:, R_AN, :3] - skeleton_seq[:, R_KN, :3]
        r_leg_mat = KinematicRetargeter._construct_basis(r_leg_fwd, r_leg_ref)
        quats["right_upper_leg"] = KinematicRetargeter._matrix_to_quaternion(r_leg_mat)

        return quats

    @staticmethod
    def extract_joint_angles(skeleton_seq: np.ndarray) -> dict:
        """Legacy 1D Flexion Extractor (Maintained for backwards compatibility)."""
        if skeleton_seq.shape[0] == 0: return {}
        angles = {}
        angles["left_elbow_flexion"] = KinematicRetargeter._angle_between_vectors(
            skeleton_seq[:, 11, :3] - skeleton_seq[:, 13, :3], skeleton_seq[:, 15, :3] - skeleton_seq[:, 13, :3])
        angles["right_elbow_flexion"] = KinematicRetargeter._angle_between_vectors(
            skeleton_seq[:, 12, :3] - skeleton_seq[:, 14, :3], skeleton_seq[:, 16, :3] - skeleton_seq[:, 14, :3])
        angles["left_knee_flexion"] = KinematicRetargeter._angle_between_vectors(
            skeleton_seq[:, 23, :3] - skeleton_seq[:, 25, :3], skeleton_seq[:, 27, :3] - skeleton_seq[:, 25, :3])
        angles["right_knee_flexion"] = KinematicRetargeter._angle_between_vectors(
            skeleton_seq[:, 24, :3] - skeleton_seq[:, 26, :3], skeleton_seq[:, 28, :3] - skeleton_seq[:, 26, :3])
        return angles