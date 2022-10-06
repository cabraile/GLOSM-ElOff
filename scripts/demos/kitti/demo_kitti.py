# 3rd party packages
import os
import sys
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import contextily as cx
import utm

import torch
import rasterio
import rasterio.plot

import pykitti
from pykitti.utils import OxtsPacket

# Importing GLOSM
MODULES_PATH =os.path.realpath( 
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), # root/scripts/demos/kitti/
        os.path.pardir, # root/scripts/demos/
        os.path.pardir  # root/scripts/
    )
)
sys.path.append(MODULES_PATH)
from glosm.odometry import GPSOdometryProvider
from glosm.mcl import MCL
from glosm.draw import draw_pose_2d, draw_navigable_area, draw_traffic_signals
from glosm.map_manager import load_map_layers
from glosm.ekf import EKF3D

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to the dataset root directory.")
    parser.add_argument("--date", default= "2011_09_26", help="Date of the dataset recording (YYYY_MM_DD).")
    parser.add_argument("--drive", default= "0009", help="Drive number (XXXX).")
    parser.add_argument("--dsm_path", required=True, help="Path to the digital surface model's root directory.")
    parser.add_argument("--n_frames", default= 0, type=int, help="Number of frames to be loaded.")
    parser.add_argument("--osm", required=True, help="Path where the '.osm.pbf' is stored. If not exists, will download the dataset.")
    return parser.parse_args()

def get_groundtruth_state_as_array( seq_data : OxtsPacket) -> np.ndarray:
    """Given a Kitti GPS data, convert it to a state vector."""
    easting, northing, _, _ = utm.from_latlon(seq_data.lat, seq_data.lon)
    elevation = seq_data.alt
    roll    = seq_data.roll
    pitch   = seq_data.pitch
    yaw     = seq_data.yaw
    velocity_north  = seq_data.vn
    velocity_east   = seq_data.ve
    velocity_up     = seq_data.vu
    velocity_roll   = seq_data.wx
    velocity_pitch  = seq_data.wy
    velocity_yaw    = seq_data.wz
    data_array = np.array([
        [easting], [northing], [elevation],
        [roll], [pitch], [yaw],
        [velocity_east], [velocity_north], [velocity_up],
        [velocity_roll], [velocity_pitch], [velocity_yaw]
    ])
    return data_array

def main() -> int:
    args = parse_args()
    output_prefix = f"{args.date}_drive_{args.drive}_sync"

    # Map elements
    print("Loading DSM")
    print("======================")
    start_time = time.time()
    dsm_raster = rasterio.open(os.path.abspath(args.dsm_path))
    print(f"Took {time.time()-start_time:.2f}s")

    print("Loading OSM data")
    print("======================")
    start_time = time.time()
    layers = load_map_layers(args.osm)
    print(f"Took {time.time()-start_time:.2f}s")

    # Kitti data
    data = pykitti.raw(os.path.abspath(args.dataset), args.date, args.drive)
    K_left  = data.calib.K_cam3
    K_right = data.calib.K_cam2
    T_from_lidar_to_camera_left = data.calib.T_cam3_velo
    T_from_lidar_to_camera_right= data.calib.T_cam2_velo
    T_from_imu_to_lidar         = data.calib.T_velo_imu
    
    T_from_lidar_to_imu         = np.linalg.inv(T_from_imu_to_lidar) # TODO: use closed-form formula
    T_from_left_camera_to_imu   = T_from_lidar_to_imu @ np.linalg.inv(T_from_lidar_to_camera_left)
    T_from_right_camera_to_imu  = T_from_lidar_to_imu @ np.linalg.inv(T_from_lidar_to_camera_right)

    # TODO: Use cam0 as reference - comparison with the odometry methods
    # T_from_lidar_to_base_link = data.calib.T_cam0_velo
    # T_from_imu_to_base_link = T_from_lidar_to_base_link @ T_from_imu_to_lidar

    # Detection Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).half()

    # Main loop
    print()
    print("Started loop")
    print("======================")
    initial_mean = get_groundtruth_state_as_array(data.oxts[0].packet)

    mcl = MCL()
    mcl.set_digital_surface_model_map(dsm_raster)
    mcl.set_driveable_map(layers["driveable_area"])
    mcl.set_traffic_signals_map(layers["traffic_signals"])
    mcl.set_stop_signs_map(layers["stop_signs"])
    mcl.sample(initial_mean.flatten()[[0,1,5]], np.diag([1.0,1.0, 1e-2]),n_particles=500)
    mcl.load_local_region_map( initial_mean[0], initial_mean[1], 200.0 )
    odometry_provider = GPSOdometryProvider(
        xyz=initial_mean.flatten()[:3] + np.random.multivariate_normal([0.,0.,0.], cov=np.diag([1e-1, 1e-1, 1e-2])),
        rpy=initial_mean.flatten()[3:6] + np.random.multivariate_normal([0.,0.,0.], cov=np.diag([1e-4, 1e-4, 1e-4]))
    )

    ekf_corrected = EKF3D(
        current_timestamp = 0.0, 
        initial_mean = initial_mean,
        initial_covariance = np.diag([1e-1, 1e-1, 1e-2, 1e-4, 1e-4, 1e-4, 1e-1, 1e-1, 1e-1, 1e-2, 1e-2, 1e-2])
    )

    fig, ax = plt.subplots(1,1,figsize=(15,15))

    # Stores results for exporting as CSV
    estimated_trajectory = []
    groundtruth_trajectory = []

    for seq_idx in range(1,len(data)):

        # Retrieve sequence data
        elapsed_time = (data.timestamps[seq_idx] - data.timestamps[0]).total_seconds()
        print(f"Sequence ({seq_idx}/{len(data)}) - elapsed time: {elapsed_time}s")
        print("------------------------")
        #lidar_scan  = data.get_velo(seq_idx)
        gps_and_imu_data = data.oxts[seq_idx].packet
        rgb_right, rgb_left = data.get_rgb(seq_idx) # right = cam2, left = cam3
        rgb_left = np.array(rgb_left)
        rgb_right = np.array(rgb_right)
        groundtruth_mean = get_groundtruth_state_as_array(gps_and_imu_data)

        # EKF prediction
        seq_start_time = time.time()
        # TODO: Implement true odometry estimation
        control_array = odometry_provider.step(
            xyz=groundtruth_mean.flatten()[:3] + np.random.multivariate_normal([0.,0.,0.], cov=np.diag([1e-1, 1e-1, 1e-2])),
            rpy=groundtruth_mean.flatten()[3:6] + np.random.multivariate_normal([0.,0.,0.], cov=np.diag([1e-4, 1e-4, 1e-4]))
        )
        mcl.predict(control_array[[0,1,5]], np.diag([1e-1,1e-1,1e-4]), z_offset=control_array.flatten()[2],z_variance=1e-2)
        duration = time.time() - seq_start_time
        print(f"Took {1000*duration:.0f}ms for MCL prediction")

        # Detect traffic signals
        model_detection = model(rgb_left)
        detections = model_detection.pandas().xyxy[0]
        detections = detections[detections["confidence"] > 0.7]
        detected_traffic_signal = np.any( detections["name"] == "traffic light" )
        detected_stop_sign = np.any( detections["name"] == "stop sign" )

        # MCL Update
        start_time = time.time()
        mcl.weigh_in_driveable_area()

        if detected_traffic_signal:
            mcl.weigh_traffic_signal_detection(sensitivity=0.95, false_positive_rate = 0.05)
 
        if seq_idx % 5 == 0:
            mcl.weigh_eloff()
        duration = time.time() - start_time
        print(f"Took {1000*duration:.0f}ms for MCL update")

        ekf_corrected.predict(elapsed_time, cov= np.diag([1e-1,1e-1,1e-1,1e-2,1e-2,1e-2,1e-1,1e-1,1e-1,1e-2,1e-2,1e-2]),)
        ekf_corrected.predict_pose3d(
            delta_xyz=control_array.flatten()[:3],
            delta_rpy=control_array.flatten()[3:],
            cov = np.diag([1e-1,1e-1,1e-1,1e-2,1e-2,1e-2,1e-1,1e-1,1e-1,1e-2,1e-2,1e-2])
        )

        # EKF Update
        mcl_pose2d = mcl.get_mean().flatten()
        mcl_cov2d = mcl.get_covariance()
        ekf_corrected.update_pose2d(
            x = mcl_pose2d[0],
            y = mcl_pose2d[1],
            yaw = mcl_pose2d[2],
            Q = mcl_cov2d,
            timestamp = elapsed_time
        )

        # Store trajectory
        easting,northing,elevation = groundtruth_mean.flatten()[:3]
        x,y,z = groundtruth_mean.flatten()[:3] - initial_mean.flatten()[:3]
        groundtruth_trajectory.append({
            "elapsed_time" : elapsed_time, "easting" : easting, "northing" : northing, "elevation" : elevation,  "x" : x, "y" : y, "z" : z
        })
        easting,northing,elevation = ekf_corrected.get_state().flatten()[:3]
        x,y,z = ekf_corrected.get_state().flatten()[:3] - initial_mean.flatten()[:3]
        estimated_trajectory.append({
            "elapsed_time" : elapsed_time, "easting" : easting, "northing" : northing, "elevation" : elevation,  "x" : x, "y" : y, "z" : z
        })

        # Draw using the local map
        if seq_idx % 5 == 0:
            start_time = time.time()
            x_gt, y_gt, yaw_gt = groundtruth_mean.flatten()[[0,1,5]]
            xyz_est, rpy_est, _, _ = ekf_corrected.split_state(ekf_corrected.get_state())
            x_est, y_est = xyz_est.flatten()[:2]
            yaw_est = rpy_est.flatten()[2]
            ax.cla()
            region_size_meters = 80.0
            ax.set_xlim([x_gt - region_size_meters/2.0, x_gt + region_size_meters/2.0])
            ax.set_ylim([y_gt - region_size_meters/2.0, y_gt + region_size_meters/2.0])
            cx.add_basemap(ax, crs=layers["driveable_area"].crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)
            draw_navigable_area(layers["driveable_area"], ax)
            rasterio.plot.show(dsm_raster, ax=ax, cmap="jet",alpha=1.0)
            mcl.plot(ax)
            draw_pose_2d(x_gt,y_gt, yaw_gt, ax, "Groundtruth")
            draw_pose_2d(x_est,y_est, yaw_est, ax, "Estimation")
            draw_traffic_signals(layers["traffic_signals"], ax)
            if not os.path.exists(f"results/{output_prefix}/frames"):
                os.makedirs(f"results/{output_prefix}/frames")
            plt.savefig(f"results/{output_prefix}/frames/{seq_idx:05}.png")
            duration = time.time() - start_time
            print(f"Took {1000*duration:.0f}ms for updating the visualization")
        
        print(f"Complete pipeline in the sequence took {time.time()-seq_start_time:.2f}s")
    
    print("-----------\n")
    print("Exporting video to `results/`")
    os.system(f"ffmpeg -framerate 10 -pattern_type glob -i 'results/{output_prefix}/frames/*.png' results/{output_prefix}/{output_prefix}.mp4")

    print("-----------\n")
    print(f"Exporting trajectories to `results/{output_prefix}`")
    pd.DataFrame(groundtruth_trajectory).to_csv(f"results/{output_prefix}/groundtruth_trajectory.csv",index=False)
    pd.DataFrame(estimated_trajectory).to_csv(f"results/{output_prefix}/estimated_trajectory.csv",index=False)
    return 0

if __name__=="__main__":
    sys.exit(main())