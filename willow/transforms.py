import numpy as np

class CoordinateBridge:
    """
    Bridges the standard Willow space (MediaPipe: X-Right, Y-Down, Z-Forward) 
    to industry-standard spaces for Robotics and VR/AR.
    """

    @staticmethod
    def to_ros_z_up(skeleton_seq: np.ndarray) -> np.ndarray:
        """
        Translates to ROS/Isaac Sim Standard (Z-Up, Right-Handed).
        Maps: MP_X -> ROS_X, MP_Z -> ROS_Y, -MP_Y -> ROS_Z
        """
        transformed = np.zeros_like(skeleton_seq)
        transformed[..., 0] = skeleton_seq[..., 0]   # X = X
        transformed[..., 1] = skeleton_seq[..., 2]   # Y = Depth
        transformed[..., 2] = -skeleton_seq[..., 1]  # Z = Up
        if skeleton_seq.shape[-1] == 4:
            transformed[..., 3] = skeleton_seq[..., 3]
        return transformed

    @staticmethod
    def to_unity_y_up(skeleton_seq: np.ndarray) -> np.ndarray:
        """
        Translates to Unity/Unreal Standard (Y-Up, Left-Handed).
        Maps: MP_X -> Unity_X, -MP_Y -> Unity_Y, MP_Z -> Unity_Z
        """
        transformed = np.zeros_like(skeleton_seq)
        transformed[..., 0] = skeleton_seq[..., 0]
        transformed[..., 1] = -skeleton_seq[..., 1]
        transformed[..., 2] = skeleton_seq[..., 2]
        if skeleton_seq.shape[-1] == 4:
            transformed[..., 3] = skeleton_seq[..., 3]
        return transformed