import os
import sys
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import contextily as cx
import utm

import torch
import rasterio

import pykitti
from pykitti.utils import OxtsPacket

from mapless.odometry import GPSOdometry
from mapless.mcl import MCL
from mapless.draw import draw_pose_2d, draw_navigable_area, draw_traffic_signals
from mapless.map_manager import load_map_layers

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

    # Detection Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

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
    mcl.sample(initial_mean.flatten()[[0,1,5]], np.diag([10.0,10.0, 1e-1]),n_particles=100)
    mcl.load_local_region_map( initial_mean[0], initial_mean[1], 200.0 )

    odometry = GPSOdometry(
        xyz=initial_mean.flatten()[:3] + np.random.multivariate_normal([0.,0.,0.], cov=np.diag([1e-1, 1e-1, 1e-1])),
        rpy=initial_mean.flatten()[3:6] + np.random.multivariate_normal([0.,0.,0.], cov=np.diag([1e-2, 1e-2, 1e-2]))
    )

    fig, ax = plt.subplots(1,1,figsize=(15,15))

    for seq_idx in range(1,len(data)):
        # Retrieve sequence data
        timestamp = (data.timestamps[seq_idx] - data.timestamps[0]).total_seconds()
        print(f"Sequence {seq_idx} - timestamp: {timestamp}s")
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
        control_array = odometry.step(
            xyz=groundtruth_mean.flatten()[:3] + np.random.multivariate_normal([0.,0.,0.], cov=np.diag([1e-1, 1e-1, 1e-1])),
            rpy=groundtruth_mean.flatten()[3:6] + np.random.multivariate_normal([0.,0.,0.], cov=np.diag([1e-2, 1e-2, 1e-2]))
        )
        mcl.predict(control_array[[0,1,5]], np.diag([1e-1,1e-1,1e-2]), z_offset=control_array.flatten()[2],z_variance=1e-2)
        duration = time.time() - seq_start_time
        print(f"Took {1000*duration:.0f}ms for MCL prediction")

        # Detect traffic signals
        model_detection = model(rgb_left)
        detections = model_detection.pandas().xyxy[0]
        detections = detections[detections["confidence"] > 0.7]
        detected_traffic_signal = np.any( detections["name"] == "traffic light" )
        detected_stop_sign = np.any( detections["name"] == "stop sign" )
        if detected_stop_sign or detected_traffic_signal:
            model_detection.show()

        # MCL Update
        start_time = time.time()
        mcl.weigh_in_driveable_area()

        if detected_traffic_signal:
            mcl.weigh_traffic_signal_detection(sensitivity=0.95, false_positive_rate = 0.05)
        duration = time.time() - start_time
        print(f"Took {1000*duration:.0f}ms for MCL update")
 
        if seq_idx % 10 == 0.0:
            print("here")
            mcl.weigh_eloff()

        # Draw using the local map
        if seq_idx % 5 == 0.0:
            start_time = time.time()
            x_gt, y_gt, yaw_gt = groundtruth_mean.flatten()[[0,1,5]]
            ax.cla()
            draw_pose_2d(x_gt,y_gt, yaw_gt, ax, "Groundtruth")
            mcl.plot(ax)
            odometry.plot(ax)
            cx.add_basemap(ax, crs=layers["driveable_area"].crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)
            draw_navigable_area(layers["driveable_area"], ax)
            draw_traffic_signals(layers["traffic_signals"], ax)
            #TODO: solve conflicts for displaying output!
            plt.savefig(f"output/{seq_idx}.png")
            plt.pause(0.01)
            duration = time.time() - start_time
            print(f"Took {1000*duration:.0f}ms for updating the visualization")

        print(f"Complete pipeline in the sequence took {time.time()-seq_start_time:.2f}s")
    plt.show()
    return 0

if __name__=="__main__":
    sys.exit(main())