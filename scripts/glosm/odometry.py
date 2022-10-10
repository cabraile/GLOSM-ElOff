from typing import Optional
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from glosm.rotation import rotation_matrix_from_euler_angles
from glosm.draw import draw_pose_2d

class InertialOdometry:
    # TODO: for now ignores xyz
    def __init__(
        self, xyz : np.ndarray, rpy : np.ndarray, xyz_dot : np.ndarray, frequency : float
    ) -> None:
        self.xyz = xyz.reshape(3,1) # w.r.t. world
        self.rpy = rpy.reshape(3,1) # w.r.t. world
        self.xyz_dot = xyz_dot.reshape(3,1) # w.r.t. world
        self.frequency = frequency
        self.delta_t = 1 / self.frequency

    def pose_as_transformation_matrix(self) -> np.ndarray:
        R_from_frame_to_world = rotation_matrix_from_euler_angles(*self.rpy.flatten())
        T_from_frame_to_world = np.eye(4)
        T_from_frame_to_world[:3,:3] = R_from_frame_to_world
        T_from_frame_to_world[:3, 3] = self.xyz.flatten()
        return T_from_frame_to_world

    def step(self, acceleration_imu : np.array, angular_velocity : np.array) -> np.ndarray :
        delta_t = self.delta_t

        # Get current transformation matrix
        T_from_imu_prev_to_world = self.pose_as_transformation_matrix()

        # Convert IMU to world frame
        R_from_imu_prev_to_world = rotation_matrix_from_euler_angles(*self.rpy.flatten())
        a_world = R_from_imu_prev_to_world @ acceleration_imu.reshape(3,1)

        # Update position
        delta_position_world = self.xyz_dot * delta_t + a_world * (delta_t**2.0)/2.0
        self.xyz = self.xyz + delta_position_world

        # Update velocity
        self.xyz_dot = self.xyz_dot + a_world * delta_t
        
        # Update orientation
        delta_orientation = angular_velocity * delta_t
        R_from_imu_curr_to_imu_prev = rotation_matrix_from_euler_angles(*delta_orientation.flatten())
        R_from_imu_curr_to_world = R_from_imu_prev_to_world @ R_from_imu_curr_to_imu_prev
        self.rpy = Rotation.from_matrix(R_from_imu_curr_to_world).as_euler("xyz",degrees=False).reshape(3,1)

        # Get current transformation matrix
        T_from_imu_curr_to_world = self.pose_as_transformation_matrix()

        # Transform from current to prev
        T_from_world_to_imu_prev = np.linalg.inv(T_from_imu_prev_to_world) # TODO: used closed-formula solution
        T_from_imu_curr_to_prev = T_from_world_to_imu_prev @ T_from_imu_curr_to_world

        # Get displacement
        return T_from_imu_curr_to_prev

    def plot(self, ax : plt.Axes) -> None:
        x, y = self.xyz[[0,1],[0,0]]
        yaw = self.rpy[2,0]
        draw_pose_2d(x,y,yaw,ax,label="Odometry")

class GPSOdometryProvider:

    def __init__(self, xyz : np.ndarray, rpy : np.ndarray ) -> None:
        self.xyz = xyz.reshape(3,1)
        self.rpy = rpy.reshape(3,1)
    
    def as_transformation_matrix(self) -> np.ndarray:
        R_from_frame_to_world = rotation_matrix_from_euler_angles(*self.rpy.flatten())
        T_from_frame_to_world = np.eye(4)
        T_from_frame_to_world[:3,:3] = R_from_frame_to_world
        T_from_frame_to_world[:3, 3] = self.xyz.flatten()
        return T_from_frame_to_world

    def plot(self, ax : plt.Axes) -> None:
        x, y = self.xyz[[0,1],[0,0]]
        yaw = self.rpy[2,0]
        draw_pose_2d(x,y,yaw,ax,label="Odometry")

    def step(self, xyz : np.ndarray, rpy : np.ndarray) -> np.ndarray:
        T_from_prev_to_world = self.as_transformation_matrix()
        self.xyz = xyz.reshape(3,1)
        self.rpy = rpy.reshape(3,1)
        T_from_curr_to_world = self.as_transformation_matrix()
        T_from_world_to_prev = np.linalg.inv(T_from_prev_to_world) # TODO: used closed-formula solution
        T_from_curr_to_prev = T_from_world_to_prev @ T_from_curr_to_world

        # Get displacement
        delta_position = T_from_curr_to_prev[:3,3].reshape(3,1)
        delta_orientation = Rotation.from_matrix(T_from_curr_to_prev[:3,:3]).as_euler("xyz",degrees=False).reshape(3,1)

        return np.vstack([delta_position, delta_orientation])