import numpy as np

class PhysicsEvaluator:
    """
    Ports the Da Vinci Backend's physics derivatives to the Edge.
    Allows real-time coaching apps to evaluate speed, acceleration, and jerk.
    """

    @staticmethod
    def calculate_derivatives(end_effector_trajectory: np.ndarray, fps: float) -> dict:
        """
        Calculates Speed (m/s), Acceleration (m/s^2), and Jerk (m/s^3) 
        for a specific 3D trajectory sequence.
        """
        dt = 1.0 / fps if fps > 0 else 0.033
        
        # 1st Derivative: Velocity
        velocity = np.gradient(end_effector_trajectory, dt, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        
        # 2nd Derivative: Acceleration
        acceleration = np.gradient(velocity, dt, axis=0)
        accel_mag = np.linalg.norm(acceleration, axis=1)
        
        # 3rd Derivative: Jerk (Smoothness penalty)
        jerk = np.gradient(acceleration, dt, axis=0)
        jerk_mag = np.linalg.norm(jerk, axis=1)

        return {
            "speed_mps": speed,
            "acceleration_mps2": accel_mag,
            "jerk_mps3": jerk_mag,
            "peak_speed": float(np.max(speed)),
            "peak_jerk": float(np.max(jerk_mag))
        }