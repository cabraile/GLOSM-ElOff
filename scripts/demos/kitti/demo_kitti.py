import os
import yaml
import sys
import time
import argparse
import tqdm
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

import contextily as cx
import utm
from scipy.spatial.transform import Rotation

import rasterio
import rasterio.plot
from rasterio.windows import from_bounds

import pykitti
from pykitti.utils  import OxtsPacket

import open3d as o3d
from slopy.odometry import Odometry as LaserOdometry

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

from glosm.odometry import KittiFileReadOdometry
from glosm.mcl          import MCLXYZYaw
from glosm.draw         import draw_pose_2d, draw_navigable_area
from glosm.map_manager  import load_map_layers
from sequences_utils    import seq_to_date_mapping, seq_to_drive_mapping, seq_to_start_idx, seq_to_end_idx

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     required=True, help="Path to the dataset root directory.")
    parser.add_argument("--sequence_id", required=True, help="Two-digits id of the Kitti sequence.")
    parser.add_argument("--dsm_path",    required=True, help="Path to the digital surface model's root directory.")
    parser.add_argument("--localization_mode",  default="glosm-eloff", choices=["glosm", "eloff", "glosm-eloff"], help="Whether to use only GLOSM, ElOff or both for estimations.")
    parser.add_argument("--odometry_name",      default="odometry", help="Name of the odometry method used. Visualization purposes only.")
    parser.add_argument("--odometry_mode",      default="slopy", choices=["slopy", "file"], help="How to estimate odometry")
    parser.add_argument("--odometry_file_path", default="", help="The path to the file that contain the odometry estimates. Mandatory if `odometry_mode='file'`.")
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

class DemoManager:

    def __init__(self, args : argparse.Namespace) -> None:
        self.transforms = dict()

        # Loads the arguments
        self.odometry_name      = args.odometry_name
        self.odometry_mode      = args.odometry_mode
        self.localization_mode  = args.localization_mode
        self.sequence_id        = args.sequence_id
        self.odometry_file_path = args.odometry_file_path
        self.seq_start          = seq_to_start_idx[self.sequence_id]
        self.seq_end            = seq_to_end_idx[self.sequence_id]

        # Loads the configuration file
        config_path = os.path.join(PROJECT_PATH,"cfg", "demo.yaml")
        with open(config_path,"r") as cfg_file:
            self.config = yaml.load(cfg_file,yaml.FullLoader)

        self.output_prefix = f"{self.odometry_name}+{self.localization_mode}"
        self.video_output_dir = f"results/videos/{self.output_prefix}"
        self.trajectories_output_dir = f"results/trajectories/{self.output_prefix}"
        os.system(f"mkdir -p {self.video_output_dir}")
        os.system(f"mkdir -p {self.trajectories_output_dir}/corrected")
        os.system(f"mkdir -p {self.trajectories_output_dir}/uncorrected")

        # Load map layers
        print("Loading Map Layers")
        print("======================")
        self.dsm_raster = rasterio.open( os.path.abspath(args.dsm_path) )
        if "glosm" in self.localization_mode:
            self.map_layers = load_map_layers(self.config["cache_dir"])

        # Load kitti data
        print("Loading Kitti Elements")
        print("======================")
        kitti_date  = seq_to_date_mapping[self.sequence_id]
        kitti_drive = seq_to_drive_mapping[self.sequence_id]
        self.kitti_dataloader = pykitti.raw(os.path.abspath(args.dataset), kitti_date, kitti_drive)

        # Register kitti frames
        self.register_transform(self.kitti_dataloader.calib.T_velo_imu, source_frame = "imu", target_frame = "lidar")
        self.register_transform(self.kitti_dataloader.calib.T_cam0_velo, source_frame = "lidar", target_frame = "cam0")
        
        T_from_imu_to_cam0 = self.transforms["lidar"]["cam0"] @ self.transforms["imu"]["lidar"]
        self.register_transform(T_from_imu_to_cam0, source_frame = "imu", target_frame = "cam0")

        initial_mean = get_groundtruth_state_as_array(self.kitti_dataloader.oxts[self.seq_start].packet)
        T_from_imu_init_to_utm = np.eye(4)
        T_from_imu_init_to_utm[:3,:3] = Rotation.from_euler(
            "xyz",
            angles=initial_mean.flatten()[3:6], 
            degrees=False
        ).as_matrix()
        T_from_imu_init_to_utm[:3,3] = initial_mean.flatten()[0:3]
        self.register_transform(T_from_imu_init_to_utm, source_frame = "imu_init", target_frame = "utm")

        print("Loading odometry source")
        print("======================")
        self.load_odometry()

        print("Initializing odometry and MCL modules")
        print("======================")
        
        # Initializes the particle filter
        self.mcl = MCLXYZYaw()
        if "eloff" in self.localization_mode:
            self.mcl.set_digital_surface_model_map(self.dsm_raster)
        if "glosm" in self.localization_mode:
            self.mcl.set_driveable_map(self.map_layers["driveable_area"])
            self.mcl.set_traffic_signals_map(self.map_layers["traffic_signals"])
            self.mcl.set_stop_signs_map(self.map_layers["stop_signs"])
            self.mcl.load_local_region_map( initial_mean[0], initial_mean[1], 200.0 )
        self.mcl.sample(
            initial_mean.flatten()[[0,1,2,5]], 
            np.diag([0.01,0.01,0.01, 1e-4]), 
            n_particles=self.config["n_particles"]
        )

        # Stores results

        self.trajectories = {
            "odometry" : [ list(np.eye(3,4).flatten()) ],
            "estimated" : [ list(np.eye(3,4).flatten()) ]
        }
        self.benchmark_time_lists = {
            "prediction" : [],
            "glosm" : [],
            "eloff" : []
        }

    def register_transform(self, T : np.ndarray, source_frame : str, target_frame : int) -> None:
        # Stores the forward transform
        if source_frame not in self.transforms:
            self.transforms[source_frame] = dict()
        self.transforms[source_frame][target_frame] = T
        
        # Stores the inverse transform
        if target_frame not in self.transforms:
            self.transforms[target_frame] = dict()
        self.transforms[target_frame][source_frame] = np.linalg.inv(T) # TODO: use closed formula

    def load_odometry(self) -> None:
        if self.odometry_mode == "slopy":
            print("> Starting SLOPY")
            self.odometry = LaserOdometry(
                voxel_size=0.25, 
                distance_threshold=3.5, 
                frequency=10.0,
                mode="point-to-plane"
            )
            # Register first scan - identity
            lidar_pcd = scan_array_to_pointcloud(self.kitti_dataloader.get_velo(self.seq_start))
            self.odometry.register(lidar_pcd)
            
        elif self.odometry_mode == "file":
            print("> Loading odometry file")
            print(f"> Loading from sequence idx {self.seq_start}")
            initial_mean = get_groundtruth_state_as_array(self.kitti_dataloader.oxts[self.seq_start].packet)
            self.odometry = KittiFileReadOdometry(
                self.odometry_file_path,
                xyz=initial_mean.flatten()[[0,1,2]],
                rpy=initial_mean.flatten()[[3,4,5]],
                T_from_imu_to_cam0=self.transforms["imu"]["cam0"]
            )
            assert len(self.odometry.trajectory_imu) == self.seq_end - self.seq_start + 1, \
                f"Error: odometry loaded from file contains {len(self.odometry.trajectory_imu)} elements: defined sequence should contain {seq_end - seq_start + 1}!"
            print("> Done")

    def spin(self) -> None:
        print()
        print("Started loop")
        print("======================")
        self.last_eloff_update_timestamp = self.kitti_dataloader.timestamps[self.seq_start].timestamp()
        self.last_glosm_update_timestamp = self.kitti_dataloader.timestamps[self.seq_start].timestamp()
        for seq_idx in tqdm.tqdm(range(self.seq_start+1, self.seq_end+1)):
            self.iterate(seq_idx)
        self.output_results()

        print("\n-----------\n")
        print("Time statistics")
        print("-----------\n")
        print(f"Prediction: {np.average(self.benchmark_time_lists['prediction'])*1000:.2f}ms (stddev: {np.std(self.benchmark_time_lists['prediction'])*1000:.2f}ms)")
        print(f"GLOSM updates: {np.average(self.benchmark_time_lists['glosm'])*1000:.2f}ms (stddev: {np.std(self.benchmark_time_lists['glosm'])*1000:.2f}s)")
        print(f"ELOFF updates: {np.average(self.benchmark_time_lists['eloff'])*1000:.2f}ms (stddev: {np.std(self.benchmark_time_lists['eloff'])*1000:.2f}ms)")

    def output_results(self) -> None:
        print("-----------\n")
        if self.config["video"]["record_trajectory"]:
            print(f"Exporting video to `{self.video_output_dir}`")
            os.system(f"ffmpeg -framerate {self.config['video']['output_video_fps']} -pattern_type glob -i '{self.video_output_dir}/frames/*.png' {self.video_output_dir}/{self.sequence_id}.mp4")
            os.system(f"rm -rf '{self.video_output_dir}/frames'")

        print("-----------\n")
        print(f"Exporting trajectories to `{self.trajectories_output_dir}`")
        with open(f"{self.trajectories_output_dir}/corrected/{self.sequence_id}.txt","w") as corrected_seq_file:
            for transform in self.trajectories["estimated"]:
                tokens = list(map(str,transform))
                output_line = " ".join(tokens) + "\n"
                corrected_seq_file.write(output_line)

        with open(f"{self.trajectories_output_dir}/uncorrected/{self.sequence_id}.txt","w") as uncorrected_seq_file:
            for transform in self.trajectories["odometry"]:
                tokens = list(map(str,transform))
                output_line = " ".join(tokens) + "\n"
                uncorrected_seq_file.write(output_line)

    def iterate(self, seq_idx : int) -> None:
        # Load input data
        rpy_imu_prev = get_groundtruth_state_as_array(self.kitti_dataloader.oxts[seq_idx-1].packet).flatten()[3:6]

        # Filter
        T_from_imu_curr_to_imu_prev = self.get_odometry_between_pose(seq_idx)
        self.predict(seq_idx, T_from_imu_curr_to_imu_prev, rpy_imu_prev)
        self.update(seq_idx)

        # Update
        self.store_current_state()
        if self.config["video"]["record_trajectory"] and ( (seq_idx-1) % self.config["video"]["store_every_n_frames"]) == 0:
            self.draw(seq_idx)

    def get_odometry_between_pose(self, seq_idx : int) -> np.ndarray:
        if self.odometry_mode == "slopy":
            lidar_scan  = self.kitti_dataloader.get_velo(seq_idx)
            lidar_pcd = scan_array_to_pointcloud(lidar_scan)
            T_from_lidar_curr_to_lidar_prev = self.odometry.register(lidar_pcd)
            T_from_imu_curr_to_imu_prev = self.transforms["lidar"]["imu"] @ T_from_lidar_curr_to_lidar_prev @ self.transforms["imu"]["lidar"]
        
        if self.odometry_mode == "file":
            T_from_imu_curr_to_imu_prev = self.odometry.step()
        return T_from_imu_curr_to_imu_prev

    def get_odometry_transform(self) -> np.ndarray:
        if self.odometry_mode == "slopy":
            T_from_lidar_to_lidar_init = self.odometry.get_transform_from_frame_to_init()
            T_from_imu_to_imu_init = self.transforms["lidar"]["imu"] @ T_from_lidar_to_lidar_init @ self.transforms["imu"]["lidar"]
            T_from_imu_to_utm = self.transforms["imu_init"]["utm"] @ T_from_imu_to_imu_init
        elif self.odometry_mode == "file":
            T_from_imu_to_utm = self.odometry.get_transform_from_frame_to_init()
        else:
            raise Exception("Odometry mode not among the options 'slopy' or 'file'")
            
        return T_from_imu_to_utm

    def predict(
        self, 
        seq_idx : int, 
        T_from_imu_curr_to_imu_prev : np.ndarray, 
        rpy_imu_prev : np.ndarray
    ) -> None:
        pose_offset_dict = split_transform(T_from_imu_curr_to_imu_prev)

        # Z displacement must be provided in world coordinates
        R_from_imu_prev_to_utm = Rotation.from_euler(
            seq="xyz", angles=rpy_imu_prev, degrees=False
        ).as_matrix()
        _, _, delta_z_utm = R_from_imu_prev_to_utm @ T_from_imu_curr_to_imu_prev[:3,3]
        pose_offset_dict["z"] = delta_z_utm
        
        control_array = np.array([ pose_offset_dict[k] for k in ["x", "y", "z", "roll", "pitch", "yaw"] ])
        
        # Compute position-related variance
        position_std = 0.33
        position_var = position_std ** 2.0

        # Compute orientation-related variance
        orientation_std = 0.01
        orientation_var = orientation_std ** 2.0
        start_time = time.time()
        self.mcl.predict(
            delta_xyz = control_array[[0,1,2]],
            delta_yaw = control_array[5],
            position_covariance  = np.diag(
                [
                    position_var,
                    position_var,
                    self.config["eloff"]["dsm_z_stddev"] ** 2.0
                ]
            ), 
            orientation_variance = orientation_var
        )
        duration = time.time() - start_time
        self.benchmark_time_lists["prediction"].append(duration)

    def glosm_update(self, timestamp : float) -> None:
        delta_T_last_updated = timestamp - self.last_glosm_update_timestamp
        if (delta_T_last_updated >= self.config["glosm"]["map_correction_every_t_seconds"]):
            start_time = time.time()
            
            self.mcl.weigh_in_driveable_area(
                prob_inside_navigable_area = self.config["glosm"]["navigable_area_weight_factor"]
            )
            self.last_glosm_update_timestamp = timestamp

            duration = time.time()-start_time
            self.benchmark_time_lists["glosm"].append(duration)

    def eloff_update(self, timestamp : float) -> None:
        delta_T_last_updated = timestamp - self.last_eloff_update_timestamp 
        if delta_T_last_updated >= self.config["eloff"]["update_every_t_seconds"]:
            start_time = time.time()

            self.mcl.weigh_eloff(self.config["eloff"]["dsm_z_stddev"])
            self.last_eloff_update_timestamp = timestamp
            
            duration = time.time()-start_time
            self.benchmark_time_lists["eloff"].append(duration)

    def update(self, seq_idx : int) -> None:
        timestamp = self.kitti_dataloader.timestamps[seq_idx].timestamp()
        if "glosm" in self.localization_mode:
            self.glosm_update(timestamp)
        if "eloff" in self.localization_mode:
            self.eloff_update(timestamp)

    def draw(self, seq_idx : int) -> None:
        gps_and_imu_data = self.kitti_dataloader.oxts[seq_idx].packet
        groundtruth_mean = get_groundtruth_state_as_array(gps_and_imu_data)
        x_gt, y_gt, yaw_gt = groundtruth_mean.flatten()[[0,1,5]]

        x_est, y_est = self.mcl.get_mean().flatten()[:2]
        yaw_est = self.mcl.get_mean().flatten()[-1]
        
        odom_state = split_transform(self.get_odometry_transform())
        x_odom = odom_state["x"]
        y_odom = odom_state["y"]
        yaw_odom = odom_state["yaw"]

        region_size_meters = self.config["video"]["region_size_meters"]
        left    = x_gt - region_size_meters/2.0
        right   = x_gt + region_size_meters/2.0
        top     = y_gt + region_size_meters/2.0
        bottom  = y_gt - region_size_meters/2.0
        fig, ax = plt.subplots(1,1,figsize=(15,15))
        ax.set_xlim([left, right])
        ax.set_ylim([bottom, top])
        cx.add_basemap(ax, crs=self.dsm_raster.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)
        
        if "glosm" in self.localization_mode:
            draw_navigable_area(self.map_layers["driveable_area"], ax)
        
        self.mcl.plot(ax, draw_poses=False) # must be after the navigable area and before the traffic signals
        
        if "eloff" in self.localization_mode:
            raster_window = self.dsm_raster.read(1,window=from_bounds(left, bottom, right, top, self.dsm_raster.transform)).squeeze()
            ax.imshow(
                raster_window, cmap="jet", alpha=1.0, extent=(left,right, bottom,top),
                vmin=groundtruth_mean[2]-2.5, vmax=groundtruth_mean[2]+3.5
            )
        
        draw_pose_2d(x_gt,y_gt, yaw_gt, ax, label="Groundtruth")
        draw_pose_2d(x_est,y_est, yaw_est, ax, label="Estimation")
        draw_pose_2d(x_odom,y_odom, yaw_odom, ax, label=self.odometry_name)

        if not os.path.exists(f"{self.video_output_dir}/frames"):
            os.makedirs(f"{self.video_output_dir}/frames")
        plt.savefig(f"{self.video_output_dir}/frames/{seq_idx:05}.png")
        plt.close("all")

    def store_current_state(self) -> None:
        easting, northing, elevation = self.mcl.get_mean().flatten()[:3]
        yaw = self.mcl.get_mean().flatten()[-1]
        T_from_imu_to_utm = np.eye(4)
        T_from_imu_to_utm[:3,:3] = Rotation.from_euler("xyz",angles=(0.0,0.0,yaw)).as_matrix()
        T_from_imu_to_utm[:3,3] = [easting, northing, elevation]
        T_from_cam0_to_cam0_init = self.transforms["imu"]["cam0"] @ self.transforms["utm"]["imu_init"] @ T_from_imu_to_utm @ self.transforms["cam0"]["imu"]
        self.trajectories["estimated"].append(list(T_from_cam0_to_cam0_init[:3].flatten()))

        T_from_imu_to_utm = self.get_odometry_transform()
        T_from_cam0_to_cam0_init = self.transforms["imu"]["cam0"] @ self.transforms["utm"]["imu_init"] @ T_from_imu_to_utm @ self.transforms["cam0"]["imu"]
        self.trajectories["odometry"].append(list(T_from_cam0_to_cam0_init[:3].flatten()))

def main() -> int:
    args = parse_args()
    np.random.seed(0)
    os.system("clear")
    demo_manager = DemoManager(args)
    demo_manager.spin()
    return 0

if __name__=="__main__":
    sys.exit(main())