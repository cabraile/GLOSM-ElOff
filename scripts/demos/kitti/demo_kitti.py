# 3rd party packages
import os
import sys
import time
import argparse
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import contextily as cx
import utm

import torch
import rasterio
import rasterio.plot

import pykitti
from pykitti.utils  import OxtsPacket

import open3d as o3d
from slopy.odometry import Odometry as LaserOdometry
from scipy.spatial.transform import Rotation

# Importing GLOSM
MODULES_PATH =os.path.realpath( 
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), # root/scripts/demos/kitti/
        os.path.pardir, # root/scripts/demos/
        os.path.pardir  # root/scripts/
    )
)
sys.path.append(MODULES_PATH)

from glosm.mcl          import MCL
from glosm.draw         import draw_pose_2d, draw_navigable_area, draw_traffic_signals
from glosm.map_manager  import load_map_layers

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to the dataset root directory.")
    parser.add_argument("--date", default= "2011_09_26", help="Date of the dataset recording (YYYY_MM_DD).")
    parser.add_argument("--drive", default= "0009", help="Drive number (XXXX).")
    parser.add_argument("--dsm_path", required=True, help="Path to the digital surface model's root directory.")
    parser.add_argument("--n_frames", default= 0, type=int, help="Number of frames to be loaded.")
    parser.add_argument("--osm", required=True, help="Path where the '.osm.pbf' is stored. If not exists, will download the dataset.")
    parser.add_argument("--mode", default="glosm-eloff", choices=["glosm", "eloff", "glosm-eloff"], help="Whether to use only GLOSM, ElOff or both for estimations.")
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

def scan_array_to_pointcloud(scan : np.ndarray) -> o3d.geometry.PointCloud:
    """Converts the N-by-M array of the scan to a 3D point cloud.

    Drops any additional field as intensity or color.
    """
    xyz_array = scan[:,:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_array)
    return pcd

def split_transform(T) -> Dict[str, float]:
    """Splits a transformation matrix to position and orientation."""
    roll, pitch, yaw = Rotation.from_matrix(T[:3,:3]).as_euler("xyz",degrees=False)
    x, y, z = T[:3,3]
    return {"x" : x, "y" : y, "z" : z, "roll" : roll, "pitch" : pitch, "yaw" : yaw}

def main() -> int:
    args = parse_args()
    output_prefix = f"{args.mode}_{args.date}_drive_{args.drive}_sync"

    np.random.seed(0)

    # Map elements
    print("Loading DSM")
    print("======================")
    start_time = time.time()
    dsm_raster = rasterio.open(os.path.abspath(args.dsm_path))
    print(f"Took {time.time()-start_time:.2f}s")

    if "glosm" in args.mode:
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

    # Detection Model
    if "glosm" in args.mode:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).half()

    # Main loop
    initial_mean = get_groundtruth_state_as_array(data.oxts[0].packet)
    T_from_imu_init_to_utm = np.eye(4)
    T_from_imu_init_to_utm[:3,:3] = Rotation.from_euler(
        "xyz",
        angles=initial_mean.flatten()[3:6], 
        degrees=False
    ).as_matrix()
    T_from_imu_init_to_utm[:3,3] = initial_mean.flatten()[0:3]

    # Start estimation modules
    mcl = MCL()
    if "eloff" in args.mode:
        mcl.set_digital_surface_model_map(dsm_raster)
    if "glosm" in args.mode:
        mcl.set_driveable_map(layers["driveable_area"])
        mcl.set_traffic_signals_map(layers["traffic_signals"])
        mcl.set_stop_signs_map(layers["stop_signs"])
        mcl.load_local_region_map( initial_mean[0], initial_mean[1], 200.0 )
    mcl.sample(initial_mean.flatten()[[0,1,5]], np.diag([1.0,1.0, 1e-2]),n_particles=500)

    laser_odometry = LaserOdometry(voxel_size=0.25, distance_threshold=3.5, frequency=10.0)

    fig, ax = plt.subplots(1,1,figsize=(15,15))

    # Stores results for exporting as CSV
    trajectories = {
        "odometry" : [],
        "reference" : [],
        "estimated" : []
    }

    print()
    print("Started loop")
    print("======================")
    for seq_idx in range(0,len(data)):

        # LOAD DATA
        # =========================================================

        # Retrieve sequence data
        elapsed_time = (data.timestamps[seq_idx] - data.timestamps[0]).total_seconds()
        print(f"Sequence ({seq_idx}/{len(data)}) - elapsed time: {elapsed_time}s")
        print("------------------------")

        # Load input data
        gps_and_imu_data = data.oxts[seq_idx].packet
        rgb_right, rgb_left = data.get_rgb(seq_idx) # right = cam2, left = cam3
        rgb_left = np.array(rgb_left)
        rgb_right = np.array(rgb_right)
        groundtruth_mean = get_groundtruth_state_as_array(gps_and_imu_data)
        lidar_scan  = data.get_velo(seq_idx)

        # Filter laser scan points
        scan_points_distances = np.linalg.norm(lidar_scan[:,:3],axis=1) 
        keep_points_mask = (scan_points_distances < 60.0) & (scan_points_distances > 2.0)
        lidar_scan = lidar_scan[keep_points_mask,:]

        # LASER ODOMETRY
        # =========================================================

        # To Point Cloud
        lidar_pcd = scan_array_to_pointcloud(lidar_scan)

        # Odometry estimation
        seq_start_time = time.time()
        T_from_lidar_curr_to_lidar_prev = laser_odometry.register(lidar_pcd)
        T_from_imu_curr_to_imu_prev = T_from_lidar_to_imu @ T_from_lidar_curr_to_lidar_prev @ T_from_imu_to_lidar
        pose_offset_dict = split_transform(T_from_imu_curr_to_imu_prev)

        duration = time.time() - seq_start_time
        print(f"Took {1000*duration:.0f}ms for registering scan")        

        # PREDICTION
        # =========================================================

        seq_start_time = time.time()
        control_array = np.array([ pose_offset_dict[k] for k in ["x", "y", "z", "roll", "pitch", "yaw"] ])
        
        # Compute position-related variance
        error_per_meter = 0.8
        position_displacement = (pose_offset_dict["x"] ** 2.0 + pose_offset_dict["y"] ** 2.0) ** 0.5
        position_std = ( error_per_meter * position_displacement)/3.0 # ~3 stddev
        position_var = position_std ** 2.0

        # Compute orientation-related variance
        error_per_rad = 0.4
        orientation_std = ( error_per_rad * np.abs(pose_offset_dict["yaw"]))/3.0
        orientation_var = orientation_std ** 2.0

        mcl.predict(
            control_array[[0,1,5]], 
            np.diag([position_var,position_var,orientation_var]), 
            z_offset=control_array.flatten()[2],
            z_variance=1e-1
        )
        duration = time.time() - seq_start_time
        print(f"Took {1000*duration:.0f}ms for MCL prediction")
        
        # UPDATE
        # =========================================================
        
        # Detect traffic signals
        start_time = time.time()
        if "glosm" in args.mode:
            model_detection = model(rgb_left)
            detections = model_detection.pandas().xyxy[0]
            detections = detections[detections["confidence"] > 0.7]
            detected_traffic_signal = np.any( detections["name"] == "traffic light" )
            detected_stop_sign = np.any( detections["name"] == "stop sign" )

            # MCL Update (GLOSM landmarks)
            start_time = time.time()
            mcl.weigh_in_driveable_area(prob_inside_navigable_area=0.95)
            if detected_traffic_signal:
                mcl.weigh_traffic_signal_detection(sensitivity=0.95, false_positive_rate = 0.05)
        
        # MCL Update (ElOff)
        if ("eloff" in args.mode) and (seq_idx % 5 == 0):
            mcl.weigh_eloff()

        duration = time.time() - start_time
        print(f"Took {1000*duration:.0f}ms for MCL update")

        # BENCHMARK
        # =========================================================

        # Store trajectories
        # - Reference
        easting,northing,elevation = groundtruth_mean.flatten()[:3]
        x,y,z = groundtruth_mean.flatten()[:3] - initial_mean.flatten()[:3]
        roll,pitch,yaw = groundtruth_mean.flatten()[3:6]
        trajectories["reference"].append({
            "elapsed_time" : elapsed_time, 
            "easting" : easting, "northing" : northing, "elevation" : elevation,  
            "x" : x, "y" : y, "z" : z, 
            "roll" : roll, "pitch" : pitch, "yaw" : yaw
        })
        # - Estimated
        easting,northing = mcl.get_mean().flatten()[:2]
        x,y = mcl.get_mean().flatten()[:2] - initial_mean.flatten()[:2]
        yaw = mcl.get_mean().flatten()[-1]
        trajectories["estimated"].append({
            "elapsed_time" : elapsed_time, 
            "easting" : easting, "northing" : northing, "elevation" : 0.0,  
            "x" : x, "y" : y, "z" : 0.0,
            "roll" : 0.0, "pitch" : 0.0, "yaw" : yaw
        })
        # - Odometry
        T_from_lidar_to_lidar_init = laser_odometry.get_transform_from_frame_to_init()
        T_from_imu_to_imu_init = T_from_lidar_to_imu @ T_from_lidar_to_lidar_init @ T_from_imu_to_lidar
        T_from_imu_to_utm = T_from_imu_init_to_utm @ T_from_imu_to_imu_init
        easting, northing, elevation = T_from_imu_to_utm[:3,3]
        x,y,z = T_from_imu_to_imu_init[:3,3]
        roll, pitch, yaw = Rotation.from_matrix(T_from_imu_to_utm[:3,:3]).as_euler("xyz",degrees=False)
        trajectories["odometry"].append({
            "elapsed_time" : elapsed_time, 
            "easting" : easting, "northing" : northing, "elevation" : elevation,  
            "x" : x, "y" : y, "z" : z,
            "roll" : roll, "pitch" : pitch, "yaw" : yaw
        })

        # Draw using the local map
        if seq_idx % 7 == 0:
            start_time = time.time()
            x_gt, y_gt, yaw_gt = groundtruth_mean.flatten()[[0,1,5]]
            x_est, y_est = mcl.get_mean().flatten()[:2]
            yaw_est = mcl.get_mean().flatten()[-1]
            x_odom = trajectories["odometry"][-1]["easting"]
            y_odom = trajectories["odometry"][-1]["northing"]
            yaw_odom = trajectories["odometry"][-1]["yaw"]

            ax.cla()
            region_size_meters = 60.0
            ax.set_xlim([x_gt - region_size_meters/2.0, x_gt + region_size_meters/2.0])
            ax.set_ylim([y_gt - region_size_meters/2.0, y_gt + region_size_meters/2.0])
            cx.add_basemap(ax, crs=dsm_raster.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)
            
            if "glosm" in args.mode:
                draw_navigable_area(layers["driveable_area"], ax)
            mcl.plot(ax, draw_poses=False) # must be after the navigable area and before the traffic signals
            if "glosm" in args.mode:
                draw_traffic_signals(layers["traffic_signals"], ax)
            if "eloff" in args.mode:
                rasterio.plot.show(dsm_raster, ax=ax, cmap="jet",alpha=1.0)
            
            draw_pose_2d(x_gt,y_gt, yaw_gt, ax, label="Groundtruth")
            draw_pose_2d(x_est,y_est, yaw_est, ax, label="Estimation")
            draw_pose_2d(x_odom,y_odom, yaw_odom, ax, label="Odometry")
            if not os.path.exists(f"results/{output_prefix}/frames"):
                os.makedirs(f"results/{output_prefix}/frames")
            plt.savefig(f"results/{output_prefix}/frames/{seq_idx:05}.png")
            duration = time.time() - start_time
            print(f"Took {1000*duration:.0f}ms for updating the visualization")

        print(f"Complete pipeline in the sequence took {time.time()-seq_start_time:.2f}s")
    
    # OUTPUT
    # =========================================================
    print("-----------\n")
    print("Exporting video to `results/`")
    os.system(f"ffmpeg -framerate 5 -pattern_type glob -i 'results/{output_prefix}/frames/*.png' results/{output_prefix}/{output_prefix}.mp4")
    os.system(f"rm -rf 'results/{output_prefix}/frames'")

    print("-----------\n")
    print(f"Exporting trajectories to `results/{output_prefix}`")
    for trajectory_name, trajectory in trajectories.items():
        pd.DataFrame(trajectory).to_csv(f"results/{output_prefix}/{trajectory_name}_trajectory.csv",index=False)
    return 0

if __name__=="__main__":
    sys.exit(main())