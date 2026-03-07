import numpy as np

class CoordinateBridge:
    """
    Bridges the standard Willow coordinate space to industry-standard spaces for Robotics and VR/AR.
    
    Source Space (Willow/MediaPipe):
    - Origin: Hip Center (usually)
    - X: Right (+X is Right)
    - Y: Down (+Y is Down)
    - Z: Forward (+Z is Camera Plane/Depth)
    - Handedness: Right-Handed
    """

    @staticmethod
    def to_ros_z_up(skeleton_seq: np.ndarray) -> np.ndarray:
        """
        Translates to ROS / Isaac Sim Standard (Right-Handed, Z-Up).
        
        Mapping Logic:
        - Willow Z (Forward) -> ROS X (Forward)
        - Willow X (Right)   -> ROS -Y (Left) -- Corrected for standard "Forward-Left-Up" frame
        - Willow Y (Down)    -> ROS -Z (Up)
        
        :param skeleton_seq: Input array (Frames, Points, 3)
        :return: Transformed array (Frames, Points, 3)
        """
        transformed = np.zeros_like(skeleton_seq)
        
        # ROS X is Forward (Depth)
        transformed[..., 0] = skeleton_seq[..., 2]
        
        # ROS Y is Left (Willow X is Right, so flip sign)
        transformed[..., 1] = -skeleton_seq[..., 0]
        
        # ROS Z is Up (Willow Y is Down, so flip sign)
        transformed[..., 2] = -skeleton_seq[..., 1]
        
        # Preserve Visibility score if 4D
        if skeleton_seq.shape[-1] == 4:
            transformed[..., 3] = skeleton_seq[..., 3]
            
        return transformed

    @staticmethod
    def to_unity_y_up(skeleton_seq: np.ndarray) -> np.ndarray:
        """
        Translates to Unity / Unreal Standard (Left-Handed, Y-Up).
        
        Mapping Logic:
        - Willow X (Right)   -> Unity X (Right)
        - Willow Y (Down)    -> Unity -Y (Up)
        - Willow Z (Forward) -> Unity Z (Forward)
        
        :param skeleton_seq: Input array (Frames, Points, 3)
        :return: Transformed array (Frames, Points, 3)
        """
        transformed = np.zeros_like(skeleton_seq)
        
        # Unity X (Right)
        transformed[..., 0] = skeleton_seq[..., 0]
        
        # Unity Y (Up) - Invert Down
        transformed[..., 1] = -skeleton_seq[..., 1]
        
        # Unity Z (Forward)
        transformed[..., 2] = skeleton_seq[..., 2]
        
        # Preserve Visibility score if 4D
        if skeleton_seq.shape[-1] == 4:
            transformed[..., 3] = skeleton_seq[..., 3]
            
        return transformed