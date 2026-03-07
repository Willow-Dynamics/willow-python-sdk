import numpy as np

class PhysicsEvaluator:
    """
    Ports the Da Vinci Backend's physics derivatives to the Edge.
    Allows real-time coaching apps to evaluate speed, acceleration, and jerk
    without requiring a round-trip to the cloud.
    """

    @staticmethod
    def calculate_derivatives(end_effector_trajectory: np.ndarray, fps: float) -> dict:
        """
        Calculates Speed (m/s), Acceleration (m/s^2), and Jerk (m/s^3) 
        for a specific 3D trajectory sequence.
        
        :param end_effector_trajectory: (Frames, 3) array [x, y, z] in METERS.
        :param fps: Frames Per Second of the input source.
        :return: Dictionary of physics arrays and peak scalars.
        """
        # We need at least 3 frames to calculate a 3rd derivative (Jerk)
        if end_effector_trajectory.shape[0] < 3:
            return {
                "speed_mps": np.zeros(1), 
                "acceleration_mps2": np.zeros(1), 
                "jerk_mps3": np.zeros(1), 
                "peak_speed": 0.0, 
                "peak_jerk": 0.0
            }

        dt = 1.0 / fps if fps > 0 else 0.033
        
        # 1st Derivative: Velocity (d/dt)
        velocity = np.gradient(end_effector_trajectory, dt, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        
        # 2nd Derivative: Acceleration (d^2/dt^2)
        acceleration = np.gradient(velocity, dt, axis=0)
        accel_mag = np.linalg.norm(acceleration, axis=1)
        
        # 3rd Derivative: Jerk (d^3/dt^3) - Measure of Smoothness
        jerk = np.gradient(acceleration, dt, axis=0)
        jerk_mag = np.linalg.norm(jerk, axis=1)

        return {
            "speed_mps": speed,
            "acceleration_mps2": accel_mag,
            "jerk_mps3": jerk_mag,
            "peak_speed": float(np.max(speed)),
            "peak_jerk": float(np.max(jerk_mag))
        }