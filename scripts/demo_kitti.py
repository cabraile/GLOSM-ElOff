import os
import sys
import time
import argparse
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from pyproj import CRS
import contextily as cx
import geopandas as gpd
import utm
import pyrosm #https://pyrosm.readthedocs.io/en/latest/basics.html

import torch

import pykitti
from pykitti.utils import OxtsPacket

from mapless.odometry import GPSOdometry
from mapless.mcl import MCL
from mapless.draw import draw_pose_2d, draw_navigable_area, draw_traffic_signals

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to the dataset root directory.")
    parser.add_argument("--date", default= "2011_09_26", help="Date of the dataset recording (YYYY_MM_DD).")
    parser.add_argument("--drive", default= "0009", help="Drive number (XXXX).")
    parser.add_argument("--n_frames", default= 0, type=int, help="Number of frames to be loaded.")
    parser.add_argument("--osm", required=True, help="Path where the '.osm.pbf' is stored. If not exists, will download the dataset.")
    return parser.parse_args()

def get_utm_crs_from_lat_lon( latitude : float, longitude : float ) -> CRS:
    _,_, zone_number, zone_letter = utm.from_latlon(latitude, longitude)
    hemisphere = "north" if zone_letter.upper() >= "N" else "south"
    crs = CRS.from_string(f"+proj=utm +zone={zone_number} +{hemisphere}")
    return crs

def project_geodataframe_to_best_utm(gdf : gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, CRS]:
    # Find the centroid of the bounds
    bounds = gdf.bounds
    centroid_longitude = ( (bounds["minx"] + bounds["maxx"])/2 ).iloc[0]
    centroid_latitude = ( (bounds["miny"] + bounds["maxy"])/2 ).iloc[0]
    # Get projection
    utm_crs = get_utm_crs_from_lat_lon(centroid_latitude, centroid_longitude)
    # Project and return
    return gdf.to_crs(utm_crs), utm_crs

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

def load_road_network(osm_dir : str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Returns
    ------------
    drive_network:
        The driveable area polygons' GeoDataFrame.
    traffic_signals:
        The traffic signals' points indicated on the map as a GeoDataFrame.
    """
    road_network_path = os.path.join(osm_dir,f"road_network_utm.shp")
    traffic_signals_path = os.path.join(osm_dir,f"traffic_signals_utm.shp")

    # Loads the base map if any of the necessary elements were cached before
    if not os.path.exists(road_network_path) or not os.path.exists(traffic_signals_path):
        print("> Loading .osm.pbm")
        map_file_path = pyrosm.get_data("karlsruhe", directory=os.path.abspath(osm_dir))
        osm_map = pyrosm.OSM(map_file_path)
    
    # Loads the road the driveabe area
    if not os.path.exists(road_network_path):
        print("> Filtering road network elements")
        drive_network = osm_map.get_network(network_type="driving")
        drive_network, map_utm_crs = project_geodataframe_to_best_utm(drive_network)
        print("> Buffering navigable area - this might take long (~3 minutes)")
        # Assign one lane for lanes that were undefined
        drive_network["lanes"].fillna(value="1",inplace=True)
        # Fix wrong numbers
        drive_network["lanes"] = drive_network["lanes"].apply(lambda x : x.replace(",","."))
        drive_network["lanes"] = drive_network["lanes"].astype(float)
        drive_network = drive_network.apply( lambda row : row["geometry"].buffer(row["lanes"] * 2.5) , axis=1 )
        # Save
        print("> Saving map to file - this might take long (~5 minutes)")
        drive_network = drive_network.set_crs(map_utm_crs)
        drive_network.to_file(road_network_path)
        print(f"> Saved projected road network cache to {road_network_path}")

    # Read from cache
    else:
        print(f"> Road network cached file found at {road_network_path}")
        drive_network = gpd.read_file(road_network_path)
        map_utm_crs = drive_network.crs

    # Loads the road the driveabe area
    if not os.path.exists(traffic_signals_path):
        print("> Filtering traffic signals")
        traffic_signals = osm_map.get_data_by_custom_criteria(
            custom_filter={"highway":["traffic_signals"]}, # Keep data matching the criteria above
            filter_type="keep",
            # Do not keep nodes (point data)    
            keep_nodes=True, 
            keep_ways=False, 
            keep_relations=False
        )
        traffic_signals,_ = project_geodataframe_to_best_utm(traffic_signals)
        traffic_signals.to_file(traffic_signals_path)
        print(f"> Saved projected road network cache to {traffic_signals_path}")

    # Read from cache
    else:
        print(f"> Traffic signs cached file found at {traffic_signals_path}")
        traffic_signals = gpd.read_file(traffic_signals_path)
        map_utm_crs = drive_network.crs

    return drive_network, traffic_signals

def main() -> int:
    args = parse_args()

    # Map elements
    print("Loading OSM data")
    print("======================")
    start_time = time.time()
    drive_network, traffic_signals = load_road_network(args.osm)
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
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Main loop
    print()
    print("Started loop")
    print("======================")
    initial_mean = get_groundtruth_state_as_array(data.oxts[0].packet)
    mcl = MCL()
    mcl.set_driveable_map(drive_network)
    mcl.set_traffic_signals_map(traffic_signals)
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
        control_array = odometry.step(
            xyz=groundtruth_mean.flatten()[:3] + np.random.multivariate_normal([0.,0.,0.], cov=np.diag([1e-1, 1e-1, 1e-1])),
            rpy=groundtruth_mean.flatten()[3:6] + np.random.multivariate_normal([0.,0.,0.], cov=np.diag([1e-2, 1e-2, 1e-2]))
        )
        mcl.predict(control_array[[0,1,5]], np.diag([1e-1,1e-1,1e-2]))
        duration = time.time() - seq_start_time
        print(f"Took {1000*duration:.0f}ms for MCL prediction")

        # Detect traffic signals
        detections = model(rgb_left).pandas().xyxy[0]
        detections = detections[detections["confidence"] > 0.7]
        detected_traffic_signal = np.any( detections["name"] == "traffic light" )

        # EKF Update
        start_time = time.time()
        mcl.weigh_in_driveable_area()

        if detected_traffic_signal:
            mcl.weigh_traffic_signal_detection(sensitivity=0.95, false_positive_rate = 0.05)
        duration = time.time() - start_time
        print(f"Took {1000*duration:.0f}ms for MCL update")

        # Correction using the local map
        if seq_idx % 10 == 0.0:
            start_time = time.time()
            x_gt, y_gt, yaw_gt = groundtruth_mean.flatten()[[0,1,5]]
            ax.cla()
            draw_pose_2d(x_gt,y_gt, yaw_gt, ax, "Groundtruth")
            mcl.plot(ax)
            odometry.plot(ax)
            cx.add_basemap(ax, crs=drive_network.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)
            draw_navigable_area(drive_network, ax)
            draw_traffic_signals(traffic_signals, ax)
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