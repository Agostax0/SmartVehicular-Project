import cv2
import numpy as np

class TrajectoryKalmanFilter:
    def __init__(self, dt=0.05, measurement_var=0.05, acceleration_var=9.0):
        """
        Initialize Kalman's 2D filter to predict the trajectory of a pedestrian.
        
        Args:
            dt (float): Delta time (in seconds) between frames. 
                        Defaults at 0.05s when the simulation is running at 20 FPS.
        """
        self.dt = dt
        self.measurement_var = measurement_var
        self.acceleration_var = acceleration_var
        
        # Initialize Kalman's filter
        # 4 states [X, Z, Vx, Vz]
        # 2 measurements [X, Z]
        # 2 control variables [V_ego_x, V_ego_z]
        self.kf = cv2.KalmanFilter(4, 2, 2)
        
        # 2. Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], np.float32)
        
        # 4. Measurement noise matrix
        self.kf.measurementNoiseCov = np.array([
            [self.measurement_var, 0.0],
            [0.0, self.measurement_var]
        ], np.float32)
        
        # 5. Initial error
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
        
        # 6. Matrices that depend on dt
        self._set_dt(dt)
        
        self.initialized = False
        self.update_count = 0

    def _set_dt(self, dt):
        # ponytail: clamp long pauses; reset the filter if real stalls matter.
        self.dt = float(max(1e-3, min(dt, 0.25)))
        dt2 = self.dt * self.dt
        dt3 = dt2 * self.dt
        dt4 = dt2 * dt2

        # 1. Transition matrix
        self.kf.transitionMatrix = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], np.float32)

        # 3. Process noise matrix, constant-velocity model with acceleration noise
        q = self.acceleration_var
        self.kf.processNoiseCov = np.array([
            [dt4 / 4.0 * q, 0.0, dt3 / 2.0 * q, 0.0],
            [0.0, dt4 / 4.0 * q, 0.0, dt3 / 2.0 * q],
            [dt3 / 2.0 * q, 0.0, dt2 * q, 0.0],
            [0.0, dt3 / 2.0 * q, 0.0, dt2 * q]
        ], np.float32)

        # 6. Control matrix
        self.kf.controlMatrix = np.array([
            [-self.dt, 0.0],
            [0.0, -self.dt],
            [0.0, 0.0],
            [0.0, 0.0]
        ], np.float32)

    def predict_update(self, measured_x, measured_z, ego_vx=0.0, ego_vz=0.0, dt=None):
        """
        Predicts and Corrects the Kalman's filter trajectory.
        
        Args:
            measured_x (float): Pedestrian's horizontal offset at the current frame (meters).
            measured_z (float): Pedestrian's depth distance at the current frame (meters).
            ego_vx (float): Vehicle's horizontal velocity (m/s).
            ego_vz (float): Vehicle's approaching velocity (m/s).
            dt (float): Optional real elapsed time since the previous update.
            
        Returns:
            tuple: (est_x, est_z, vel_x, vel_z)
                   est_x: Predicted pedestrian's horizontal coordinate (relative).
                   est_z: Predicted pedestrian's depth coordinate (relative).
                   vel_x: Predicted vehicle's horizontal velocity (m/s) (absolute).
                   vel_z: Predicted vehicle's approaching velocity (m/s) (absolute).
        """
        if dt is not None:
            self._set_dt(dt)

        m_x = np.float32(measured_x)
        m_z = np.float32(measured_z)
        
        if not self.initialized:
            # At the first read, we set the measured position as the initial one
            # we assume no movement in this scenario.
            self.kf.statePost = np.array([[m_x], [m_z], [0.0], [0.0]], np.float32)
            self.kf.statePre = np.array([[m_x], [m_z], [0.0], [0.0]], np.float32)
            self.initialized = True
            self.update_count = 1
            return float(m_x), float(m_z), 0.0, 0.0
            
        # 1: PREDICT
        # The filter applies the new coordinates by applying the velocities and 
        # subtracting the vehicle movement (ego-motion)
        control = np.array([[np.float32(ego_vx)], [np.float32(ego_vz)]], np.float32)
        self.kf.predict(control)
        
        # 2: CORRECT (UPDATE)
        # We adjust the prediction using the real measurements from YOLO and the Depth Sensor
        measurement = np.array([[m_x], [m_z]], np.float32)
        estimated_state = self.kf.correct(measurement)
        
        # These are our estimated values
        est_x = float(estimated_state[0, 0])
        est_z = float(estimated_state[1, 0])
        vel_x = float(estimated_state[2, 0])
        vel_z = float(estimated_state[3, 0])
        self.update_count += 1
        
        return est_x, est_z, vel_x, vel_z
