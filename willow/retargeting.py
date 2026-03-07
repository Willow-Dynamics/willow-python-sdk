import numpy as np

class KinematicRetargeter:
    """
    Extracts joint angles (Euler approximations) from raw 3D point clouds.
    Required for Reinforcement Learning (RL) simulation environments where
    observations are based on joint states (angles), not Cartesian positions.
    """

    @staticmethod
    def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Calculates the angle in degrees between two lists of vectors.
        Includes safety checks for zero-length vectors to prevent NaN.
        """
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        
        # Create a mask for valid vectors (non-zero length)
        # We process only valid vectors to avoid division by zero runtime warnings
        mask = (n1 > 1e-6) & (n2 > 1e-6)
        
        angles = np.zeros(v1.shape[0], dtype=np.float32)
        
        if np.any(mask):
            dot_prod = np.sum(v1[mask] * v2[mask], axis=1)
            # Clip dot product to valid cosine range [-1, 1] to handle float precision errors
            cos_theta = np.clip(dot_prod / (n1[mask] * n2[mask]), -1.0, 1.0)
            angles[mask] = np.degrees(np.arccos(cos_theta))
            
        return angles

    @staticmethod
    def extract_joint_angles(skeleton_seq: np.ndarray) -> dict:
        """
        Returns a dictionary of critical 1D joint angle arrays over time.
        Useful for feeding observation buffers in Isaac Sim or PyBullet.
        """
        if skeleton_seq.shape[0] == 0:
            return {}

        # Willow 75-point Topology Keys
        L_SHOULDER, L_ELBOW, L_WRIST = 11, 13, 15
        R_SHOULDER, R_ELBOW, R_WRIST = 12, 14, 16
        L_HIP, L_KNEE, L_ANKLE = 23, 25, 27
        R_HIP, R_KNEE, R_ANKLE = 24, 26, 28

        angles = {}

        # 1. Elbow Flexions (Angle between Upper Arm vector and Forearm vector)
        l_upper_arm = skeleton_seq[:, L_SHOULDER, :3] - skeleton_seq[:, L_ELBOW, :3]
        l_forearm = skeleton_seq[:, L_WRIST, :3] - skeleton_seq[:, L_ELBOW, :3]
        angles["left_elbow_flexion"] = KinematicRetargeter._angle_between_vectors(l_upper_arm, l_forearm)

        r_upper_arm = skeleton_seq[:, R_SHOULDER, :3] - skeleton_seq[:, R_ELBOW, :3]
        r_forearm = skeleton_seq[:, R_WRIST, :3] - skeleton_seq[:, R_ELBOW, :3]
        angles["right_elbow_flexion"] = KinematicRetargeter._angle_between_vectors(r_upper_arm, r_forearm)

        # 2. Knee Flexions (Angle between Thigh vector and Calf vector)
        l_thigh = skeleton_seq[:, L_HIP, :3] - skeleton_seq[:, L_KNEE, :3]
        l_calf = skeleton_seq[:, L_ANKLE, :3] - skeleton_seq[:, L_KNEE, :3]
        angles["left_knee_flexion"] = KinematicRetargeter._angle_between_vectors(l_thigh, l_calf)

        r_thigh = skeleton_seq[:, R_HIP, :3] - skeleton_seq[:, R_KNEE, :3]
        r_calf = skeleton_seq[:, R_ANKLE, :3] - skeleton_seq[:, R_KNEE, :3]
        angles["right_knee_flexion"] = KinematicRetargeter._angle_between_vectors(r_thigh, r_calf)

        return angles