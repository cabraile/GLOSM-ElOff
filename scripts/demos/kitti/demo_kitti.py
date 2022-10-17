# 3rd party packages
import os
import yaml
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
from rasterio.windows import from_bounds

import pykitti
from pykitti.utils  import OxtsPacket

import open3d as o3d
from slopy.odometry import Odometry as LaserOdometry
from scipy.spatial.transform import Rotation

# Project-related path variables
MODULES_PATH =os.path.realpath( 
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), # root/scripts/demos/kitti/
        os.path.pardir, # root/scripts/demos/
        os.path.pardir  # root/scripts/
    )
)
PROJECT_PATH = os.path.realpath(
    os.path.join(
        MODULES_PATH,
        os.path.pardir  # root/
    )
)
sys.path.append(MODULES_PATH)

from glosm.mcl          import MCLXYZYaw
from glosm.draw         import draw_pose_2d, draw_navigable_area, draw_traffic_signals
from glosm.map_manager  import load_map_layers

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to the dataset root directory.")
    parser.add_argument("--date", required=True, help="Date of the dataset recording (YYYY_MM_DD).")
    parser.add_argument("--drive", required=True, help="Drive number (XXXX).")
    parser.add_argument("--dsm_path", required=True, help="Path to the digital surface model's root directory.")
    parser.add_argument("--n_frames", default= 0, type=int, help="Number of frames to be loaded.")
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

    # Filter laser scan points
    scan_points_distances = np.linalg.norm(scan[:,:3],axis=1) 
    keep_points_mask = (scan_points_distances < 120.0) & (scan_points_distances > 2.0)
    xyz_array = scan[keep_points_mask,:3]

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
    config_path = os.path.join(PROJECT_PATH,"cfg", "demo.yaml")
    with open(config_path,"r") as cfg_file:
        config = yaml.load(cfg_file,yaml.FullLoader)
    output_prefix = f"{args.mode}_{args.date}_drive_{args.drive}_sync"

    np.random.seed(0)

    # Map elements
    print("Loading DSM")
    print("======================")
    start_time = time.time()
    dsm_raster = rasterio.open(os.path.abspath(args.dsm_path))
    dsm_min = np.nanmin(dsm_raster.read())
    dsm_max = np.nanmax(dsm_raster.read())
    print(f"Took {time.time()-start_time:.2f}s")

    if "glosm" in args.mode:
        print("Loading OSM data")
        print("======================")
        start_time = time.time()
        layers = load_map_layers(config["cache_dir"])
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
    if ("glosm" in args.mode) and config["glosm"]["traffic_signals_landmarks"]["enable"]:
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
    mcl = MCLXYZYaw()
    if "eloff" in args.mode:
        mcl.set_digital_surface_model_map(dsm_raster)
    if "glosm" in args.mode:
        mcl.set_driveable_map(layers["driveable_area"])
        mcl.set_traffic_signals_map(layers["traffic_signals"])
        mcl.set_stop_signs_map(layers["stop_signs"])
        mcl.load_local_region_map( initial_mean[0], initial_mean[1], 200.0 )
    mcl.sample(
        initial_mean.flatten()[[0,1,2,5]], 
        np.diag([1.0,1.0,1.0, 1e-2]), 
        n_particles=config["n_particles"]
    )

    laser_odometry = LaserOdometry(
        voxel_size=0.25, 
        distance_threshold=3.5, 
        frequency=10.0,
        mode="point-to-plane"
    )

    # Register first scan - identity
    lidar_pcd = scan_array_to_pointcloud(data.get_velo(0))
    laser_odometry.register(lidar_pcd)

    # Stores results for exporting as CSV
    trajectories = {
        "odometry" : [],
        "reference" : [],
        "estimated" : []
    }

    print()
    print("Started loop")
    print("======================")
    for seq_idx in range(1,len(data)):

        # LOAD DATA
        # =========================================================

        # Retrieve sequence data
        elapsed_time = (data.timestamps[seq_idx] - data.timestamps[0]).total_seconds()
        timestamp = data.timestamps[seq_idx].timestamp()
        print(f"\nSequence ({seq_idx}/{len(data)}) - elapsed time: {elapsed_time}s")
        print("------------------------")

        # Load input data
        gps_and_imu_data = data.oxts[seq_idx].packet
        rgb_right, rgb_left = data.get_rgb(seq_idx) # right = cam2, left = cam3
        rgb_left = np.array(rgb_left)
        rgb_right = np.array(rgb_right)
        groundtruth_mean = get_groundtruth_state_as_array(gps_and_imu_data)
        lidar_scan  = data.get_velo(seq_idx)

        rpy_imu_prev = get_groundtruth_state_as_array(data.oxts[seq_idx-1].packet).flatten()[3:6]

        # LASER ODOMETRY
        # =========================================================

        # To Point Cloud
        lidar_pcd = scan_array_to_pointcloud(lidar_scan)

        # Odometry estimation
        seq_start_time = time.time()
        T_from_lidar_curr_to_lidar_prev = laser_odometry.register(lidar_pcd)
        T_from_imu_curr_to_imu_prev = T_from_lidar_to_imu @ T_from_lidar_curr_to_lidar_prev @ T_from_imu_to_lidar
        
        pose_offset_dict = split_transform(T_from_imu_curr_to_imu_prev)

        # Z displacement must be provided in world coordinates
        R_from_imu_prev_to_utm = Rotation.from_euler(seq="xyz", angles=rpy_imu_prev, degrees=False).as_matrix()
        _, _, delta_z_utm = R_from_imu_prev_to_utm @ T_from_imu_curr_to_imu_prev[:3,3]
        pose_offset_dict["z"] = delta_z_utm

        duration = time.time() - seq_start_time
        print(f"Took {1000*duration:.0f}ms for registering scan")        

        # PREDICTION
        # =========================================================

        seq_start_time = time.time()
        control_array = np.array([ pose_offset_dict[k] for k in ["x", "y", "z", "roll", "pitch", "yaw"] ])
        
        # Compute position-related variance
        error_per_meter = 1.0
        position_displacement = (pose_offset_dict["x"] ** 2.0 + pose_offset_dict["y"] ** 2.0) ** 0.5
        position_std = ( error_per_meter * position_displacement)/3.0 # ~3 stddev
        position_var = position_std ** 2.0

        # Compute orientation-related variance
        error_per_rad = 0.4
        orientation_std = ( error_per_rad * np.abs(pose_offset_dict["yaw"]))/3.0
        orientation_var = orientation_std ** 2.0

        mcl.predict(
            delta_xyz = control_array[[0,1,2]],
            delta_yaw = control_array[5],
            position_covariance  = np.diag(
                [
                    position_var,
                    position_var,
                    config["eloff"]["dsm_z_stddev"] ** 2.0
                ]
            ), 
            orientation_variance = orientation_var
        )
        duration = time.time() - seq_start_time
        print(f"Took {1000*duration:.0f}ms for MCL prediction")
        
        # UPDATE
        # =========================================================
        
        # Detect traffic signals
        duration_glosm = 0.0
        if "glosm" in args.mode:
            if config["glosm"]["traffic_signals_landmarks"]["enable"]:
                model_detection = model(rgb_left)
                detections = model_detection.pandas().xyxy[0]
                detections = detections[detections["confidence"] > 0.7]
                detected_traffic_signal = np.any( detections["name"] == "traffic light" )
                detected_stop_sign = np.any( detections["name"] == "stop sign" )

                start_time = time.time()
                if detected_traffic_signal:
                    mcl.weigh_traffic_signal_detection(
                        sensitivity = config["glosm"]["traffic_signals_landmarks"]["detector_sensitivity"], 
                        false_positive_rate = config["glosm"]["traffic_signals_landmarks"]["detector_false_positive_rate"],
                    )
                duration_glosm += time.time() - start_time

            # MCL Update (GLOSM landmarks)
            start_time = time.time()
            mcl.weigh_in_driveable_area(prob_inside_navigable_area=config["glosm"]["navigable_area_weight_factor"])
            duration_glosm += time.time() - start_time
        
        # MCL Update (ElOff)
        start_time = time.time()
        if ("eloff" in args.mode) and (seq_idx % config["eloff"]["update_every_n_frames"] == 0):
            mcl.weigh_eloff(config["eloff"]["dsm_z_stddev"])
        duration_eloff = time.time() - start_time

        duration = duration_glosm + duration_eloff
        print(f"Took {1000*duration:.0f}ms for MCL update")

        # BENCHMARK
        # =========================================================

        # Store trajectories
        # - Reference
        easting,northing,elevation = groundtruth_mean.flatten()[:3]
        x,y,z = groundtruth_mean.flatten()[:3] - initial_mean.flatten()[:3]
        roll,pitch,yaw = groundtruth_mean.flatten()[3:6]
        qx,qy,qz,qw = Rotation.from_euler("xyz",angles=(roll,pitch,yaw)).as_quat()
        trajectories["reference"].append({
            "timestamp" : timestamp,"elapsed_time" : elapsed_time,
            "easting" : easting, "northing" : northing, "elevation" : elevation,  
            "qx" : qx, "qy" : qy, "qz" : qz, "qw" : qw
        })
        # - Estimated
        easting, northing, elevation = mcl.get_mean().flatten()[:3]
        yaw = mcl.get_mean().flatten()[-1]
        qx,qy,qz,qw = Rotation.from_euler("xyz",angles=(0.0,0.0,yaw)).as_quat()
        trajectories["estimated"].append({
            "timestamp" : timestamp, 
            "easting" : easting, "northing" : northing, "elevation" : elevation,  
            "qx" : qx, "qy" : qy, "qz" : qz, "qw" : qw
        })
        # - Odometry
        T_from_lidar_to_lidar_init = laser_odometry.get_transform_from_frame_to_init()
        T_from_imu_to_imu_init = T_from_lidar_to_imu @ T_from_lidar_to_lidar_init @ T_from_imu_to_lidar
        T_from_imu_to_utm = T_from_imu_init_to_utm @ T_from_imu_to_imu_init
        easting_odom, northing_odom, elevation_odom = T_from_imu_to_utm[:3,3]
        rotation_odom = Rotation.from_matrix(T_from_imu_to_utm[:3,:3])
        _, _, yaw_odom = rotation_odom.as_euler("xyz",degrees=False)
        qx,qy,qz,qw = rotation_odom.as_quat()
        trajectories["odometry"].append({
            "timestamp" : timestamp,
            "easting" : easting_odom, "northing" : northing_odom, "elevation" : elevation_odom,  
            "qx" : qx, "qy" : qy, "qz" : qz, "qw" : qw
        })

        # Draw using the local map
        if config["video"]["record_trajectory"] and ( (seq_idx-1) % config["video"]["store_every_n_frames"]) == 0:
            start_time = time.time()
            x_gt, y_gt, yaw_gt = groundtruth_mean.flatten()[[0,1,5]]
            x_est, y_est = mcl.get_mean().flatten()[:2]
            yaw_est = mcl.get_mean().flatten()[-1]
            x_odom = easting_odom
            y_odom = northing_odom

            region_size_meters = config["video"]["region_size_meters"]
            left = x_gt - region_size_meters/2.0
            right = x_gt + region_size_meters/2.0
            top = y_gt + region_size_meters/2.0
            bottom = y_gt - region_size_meters/2.0
            fig, ax = plt.subplots(1,1,figsize=(15,15))
            ax.set_xlim([left, right])
            ax.set_ylim([bottom, top])
            cx.add_basemap(ax, crs=dsm_raster.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)
            
            if "glosm" in args.mode:
                draw_navigable_area(layers["driveable_area"], ax)
            mcl.plot(ax, draw_poses=False) # must be after the navigable area and before the traffic signals
            if "glosm" in args.mode:
                draw_traffic_signals(layers["traffic_signals"], ax)
            if "eloff" in args.mode:
                raster_window = dsm_raster.read(1,window=from_bounds(left, bottom, right, top, dsm_raster.transform)).squeeze()
                ax.imshow(
                    raster_window, cmap="jet", alpha=1.0, extent=(left,right, bottom,top),
                    vmin=groundtruth_mean[2]-2.5, vmax=groundtruth_mean[2]+3.5
                )
            
            draw_pose_2d(x_gt,y_gt, yaw_gt, ax, label="Groundtruth")
            draw_pose_2d(x_est,y_est, yaw_est, ax, label="Estimation")
            draw_pose_2d(x_odom,y_odom, yaw_odom, ax, label="Odometry")
            if not os.path.exists(f"results/{output_prefix}/frames"):
                os.makedirs(f"results/{output_prefix}/frames")
            plt.savefig(f"results/{output_prefix}/frames/{seq_idx:05}.png")
            plt.close("all")
            duration = time.time() - start_time
            print(f"Took {1000*duration:.0f}ms for updating the visualization")

        print(f"Complete pipeline in the sequence took {time.time()-seq_start_time:.2f}s")
    
    # OUTPUT
    # =========================================================
    print("-----------\n")
    if config["video"]["record_trajectory"]:
        print("Exporting video to `results/`")
        os.system(f"ffmpeg -framerate 5 -pattern_type glob -i 'results/{output_prefix}/frames/*.png' results/{output_prefix}/{output_prefix}.mp4")
        os.system(f"rm -rf 'results/{output_prefix}/frames'")

    print("-----------\n")
    print(f"Exporting trajectories to `results/{output_prefix}`")
    for trajectory_name, trajectory in trajectories.items():
        pd.DataFrame(trajectory).to_csv(
            f"results/{output_prefix}/{trajectory_name}_trajectory.csv",
            columns=["timestamp","easting","northing","elevation", "qx", "qy", "qz", "qw"],
            index=False,
            header=False,
            sep=" "
        )
    
    return 0

if __name__=="__main__":
    sys.exit(main())