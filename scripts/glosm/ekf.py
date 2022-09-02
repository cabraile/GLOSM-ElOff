from datetime import timedelta
from typing import Optional, Tuple
import numpy as np

from glosm.rotation import normalize_angle, rotation_matrix_from_euler_angles, partial_rotation_matrix
from scipy.spatial.transform import Rotation

class EKF3D:
    """Implements the 6DoF state estimation EKF.
    
    The state vector is represented by a (12,1) numpy ndarray, which estimates, respectivelly:
    1. The (x,y,z) coordinates
    2. The (roll, pitch, yaw) angles.
    3. The (x_dot, y_dot, z_dot) linear velocities.
    4. The (roll_dot, pitch_dot, yaw_dot) angular velocities.
    """
    K_STATE_DIM = 12

    def __init__(self, current_timestamp : float, initial_mean : Optional[np.ndarray] = None, initial_covariance : Optional[np.ndarray] = None) -> None:
        """
        Arguments
        ------
        current_timestamp: The float representing the number of seconds since 
            the Epoch (defined as in time.time()).
        """
        # Initialize mean
        if initial_mean is None:
            self.mean = np.zeros((EKF3D.K_STATE_DIM,1))
        else:
            self.mean = initial_mean.reshape((EKF3D.K_STATE_DIM, 1))

        # Initialize covariance
        if initial_covariance is None:
            self.covariance = np.diag([1e-18] * EKF3D.K_STATE_DIM)
        else:
            # Makes sure that the shapes match
            self.covariance = initial_covariance.reshape((EKF3D.K_STATE_DIM, EKF3D.K_STATE_DIM))

        self.timestamp = current_timestamp

    def predict(self, timestamp : float, cov : np.ndarray, acceleration_vector : Optional[np.ndarray] = None ) -> None:
        """Perform state prediction using the accelerometer data.

        Arguments
        -------
        timestamp: The float representing the number of seconds since the Epoch 
            (defined as in time.time()).
        acceleration_vector: The (ax,ay,az) array of the accelerometer.
        """
        assert cov.shape[0] == cov.shape[1] == 12, "Prediction covariance should be of size 12!"
        # Renaming for making closer to the formulas
        position, orientation, velocity, angular_velocity = EKF3D.split_state(self.mean)
        phi, theta, psi = orientation.flatten()
        if acceleration_vector is None:
            a = np.zeros((3,1), dtype=float)
        else:
            a = acceleration_vector.reshape(3,1)
    
        delta_t = timestamp - self.timestamp
        if isinstance(delta_t, timedelta):
            delta_t = delta_t.total_seconds()
        if delta_t == 0:
            return
        delta_t_sqr = delta_t ** 2

        # Build jacobian matrix
        partial_Rx = partial_rotation_matrix(roll=phi, pitch = theta, yaw=psi, partial="roll")
        partial_Ry = partial_rotation_matrix(roll=phi, pitch = theta, yaw=psi, partial="pitch")
        partial_Rz = partial_rotation_matrix(roll=phi, pitch = theta, yaw=psi, partial="yaw")

        F_xyz_rpy = np.hstack( [ partial_Rx @ a, partial_Ry @ a, partial_Rz @ a] ) * delta_t_sqr / 2.0
        F_xyz_dot_rpy = np.hstack( [ partial_Rx @ a, partial_Ry @ a, partial_Rz @ a] ) * delta_t
        F_xyz_xyz_dot = np.eye(3) * delta_t
        F_rpy_rpy_dot = np.eye(3) * delta_t
        F = np.eye(EKF3D.K_STATE_DIM)
        F[:3,3:6] = F_xyz_rpy
        F[:3,6:9] = F_xyz_xyz_dot
        F[3:6,9:] = F_rpy_rpy_dot
        F[6:9,3:6] = F_xyz_dot_rpy

        # Apply prediction to the mean
        R = rotation_matrix_from_euler_angles(roll = phi, pitch = theta, yaw = psi)
        rotated_acceleration_vector = R @ a
        position = position + delta_t * velocity + (delta_t_sqr/2.0) * rotated_acceleration_vector
        velocity = velocity + delta_t * rotated_acceleration_vector
        orientation += delta_t * angular_velocity
        
        # Normalize angles
        for i in range(3):
            orientation[i] = normalize_angle(orientation[i])

        # Apply prediction to the covariance

        covariance_pred = F @ self.covariance @ F.T + cov
        self.covariance = covariance_pred

        self.mean = np.vstack((position, orientation, velocity, angular_velocity))
        self.timestamp = timestamp

    def predict_pose3d(self, delta_xyz : np.ndarray, delta_rpy : np.ndarray , cov : np.ndarray) -> None:
        delta_roll, delta_pitch, delta_yaw = delta_rpy.flatten()
        R_from_current_to_prev = rotation_matrix_from_euler_angles(roll = delta_roll, pitch = delta_pitch, yaw = delta_yaw)
        T_from_current_to_prev = np.eye(4)
        T_from_current_to_prev[:3,:3] = R_from_current_to_prev
        T_from_current_to_prev[:3, 3] = delta_xyz.flatten()

        xyz, rpy, _, _ = self.split_state(self.mean)
        T_from_prev_to_world = np.eye(4)
        T_from_prev_to_world[:3,:3] = rotation_matrix_from_euler_angles(rpy[0,0], rpy[1,0], rpy[2,0])
        T_from_prev_to_world[:3, 3] = xyz.flatten()
        
        T_from_current_to_world = T_from_prev_to_world @ T_from_current_to_prev

        self.xyz = T_from_current_to_world[:3, 3].reshape(3,1)
        R_from_current_to_world = T_from_current_to_world[:3,:3]
        self.rpy = Rotation.from_matrix(R_from_current_to_world).as_euler("xyz", degrees=False).reshape(3,1)
        
        # TODO: DID NOT ACCOUNT FOR THE UNCERTAINTY RELATED TO THE NON-LINEARITY!
        # TODO: COMPUTE JACOBIANS
        self.covariance += cov
        
    def update_imu_orientation(self, orientation_angles_rpy : np.ndarray, angular_velocity_rpy : np.ndarray , timestamp : float, Q : np.ndarray) -> None:
        """
        Arguments
        ------
        orientations_angles_rpy: The observed orientation in roll, pitch and 
            yaw, respectivelly, from the IMU.
        angular_velocity_rpy: The observed angular velocity in roll, pitch and 
            yaw, respectivelly, from the IMU.
        timestamp: The float representing the number of seconds since the Epoch 
            (defined as in time.time()).
        Q : The 6-by-6 covariance matrix of the measurement.
        """
        # Performs a prediction for matching the current estimation of the 
        # agent with the measured.
        self.predict(timestamp=timestamp)

        H = np.zeros((6,self.K_STATE_DIM))

        # Projection of rpy
        H[:3,3:6] = np.eye(3)

        # Projection of angular velocity
        H[3:,9:] = np.eye(3)

        # Kalman update
        z = np.vstack([orientation_angles_rpy.reshape(3,1), angular_velocity_rpy.reshape(3,1)])
        P = self.covariance
        PHT = P @ (H.T)
        K = PHT @ np.linalg.inv( H @ PHT + Q )
        self.mean += K @ (z - H @ self.mean)
        self.covariance = (np.eye(EKF3D.K_STATE_DIM) - K @ H) @ P

        # Current timestamp changes!
        self.timestamp = timestamp

    def update_pose2d(self, x : float, y : float, yaw : float, timestamp : float, Q : np.ndarray) -> None:        
        # Projection considers x,y and yaw
        H = np.zeros((3,EKF3D.K_STATE_DIM))
        H[0,0] = 1 # x
        H[1,1] = 1 # y
        H[2,5] = 1 # yaw

        # Stack both elements to form measurement
        z = np.array([[x],[y],[yaw]])

        # Kalman update
        P = self.covariance
        PHT = P @ (H.T)
        K = PHT @ np.linalg.inv( H @ PHT + Q )
        self.mean += K @ (z - H @ self.mean)
        self.covariance = (np.eye(EKF3D.K_STATE_DIM) - K @ H) @ P

        # Current timestamp changes!
        self.timestamp = timestamp

    def update_pose(self, xyz : np.ndarray, rpy : np.ndarray, timestamp : float, Q : np.ndarray) -> None:
        
        # Projection considers the first 6 states (x,y,z,r,p,y)
        H = np.eye(6,12)

        # Stack both elements to form measurement
        z = np.vstack([xyz.reshape(3,1), rpy.reshape(3,1)])

        # Kalman update
        P = self.covariance
        K = ( P @ (H.T) ).dot( H @ P @ H.T + Q )
        self.mean += K @ (z - H @ self.mean)
        for i in range(3):
            self.mean[3+i] = normalize_angle(self.mean[3+i])
        self.covariance = (np.eye(EKF3D.K_STATE_DIM) - K @ H) @ P

        # Current timestamp changes!
        self.timestamp = timestamp

    def get_state(self) -> np.ndarray :
        return np.copy(self.mean)

    def get_covariance(self) -> np.ndarray:
        return np.copy(self.covariance)

    @classmethod
    def split_state(cls, state : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split the state vector into position, orientation, linear velocity, angular velocity."""
        return ( state[:3].reshape(-1,1), state[3:6].reshape(-1,1), state[6:9].reshape(-1,1), state[9:].reshape(-1,1) )