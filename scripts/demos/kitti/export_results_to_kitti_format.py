import os
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
import tqdm
import pykitti

DATASET_DIR = "/media/braile/One Touch2/Files/Acadêmico/datasets/kitti/datasets_kitti_raw_and_odometry/extracted"
DATE="2011_09_30"
DRIVE="0034"
MODE="glosm-eloff"
INPUT_FILE_PATH = f"results/2022-oct-18/{MODE}_{DATE}_drive_{DRIVE}_sync/estimated_trajectory.csv"
OUTPUT_FILE_DIR = f"results/kitti_format/"

date_drive_to_seq_mapping = {
    # Date         # Drive  # Seq  #Start seq # End seq
    "2011_10_03_drive_0027" : "00", #000000 004540
    "2011_10_03_drive_0042" : "01", #000000 001100
    "2011_10_03_drive_0034" : "02", #000000 004660
    "2011_09_26_drive_0067" : "03", #000000 000800
    "2011_09_30_drive_0016" : "04", #000000 000270
    "2011_09_30_drive_0018" : "05", #000000 002760
    "2011_09_30_drive_0020" : "06", #000000 001100
    "2011_09_30_drive_0027" : "07", #000000 001100
    "2011_09_30_drive_0028" : "08", #001100 005170
    "2011_09_30_drive_0033" : "09", #000000 001590
    "2011_09_30_drive_0034" : "10"  #000000  001200
}

seq_to_start_idx = {
    "00": 0,
    "01": 0,
    "02": 0,
    "03": 0,
    "04": 0,
    "05": 0,
    "06": 0,
    "07": 0,
    "08": 1100,
    "09": 0,
    "10": 0
}

seq_to_end_idx = {
    "00": 4540,
    "01": 1100,
    "02": 4660,
    "03": 800,
    "04": 270,
    "05": 2760,
    "06": 1100,
    "07": 1100,
    "08": 5170,
    "09": 1590,
    "10": 1200
}

def get_calibration(date : str, drive: str, dataset_dir : str) -> np.ndarray:
    """Returns the transformation matrix from the estimations' frame to the groundtruth.

    The estimations are w.r.t. the IMU, while the groundtruth is w.r.t. cam0.
    """
    kitti_data = pykitti.raw(os.path.abspath(dataset_dir), date, drive)
    T_from_lidar_to_cam0 = kitti_data.calib.T_cam0_velo
    T_from_imu_to_lidar = kitti_data.calib.T_velo_imu
    T_from_imu_to_cam0 = T_from_lidar_to_cam0 @ T_from_imu_to_lidar
    return T_from_imu_to_cam0

if __name__ == "__main__":

    input_file_path = os.path.abspath(INPUT_FILE_PATH)
    # Retrieve transformation from the frames of the estimation to the groundtruth
    T_from_imu_to_cam0 = get_calibration(DATE, DRIVE, DATASET_DIR)
    T_from_cam0_to_imu = np.linalg.inv(T_from_imu_to_cam0)
    T_from_utm_to_imu_init = None

    # Retrieve the start and end indices of the sequences
    date_drive_str = f"{DATE}_drive_{DRIVE}"
    sequence_id = date_drive_to_seq_mapping[date_drive_str]
    sequence_start_idx = seq_to_start_idx[sequence_id]
    sequence_end_idx = seq_to_end_idx[sequence_id]

    with open(input_file_path, "r") as input_file:
        out_file_path = os.path.join(os.path.abspath(OUTPUT_FILE_DIR), f"{MODE}_{sequence_id}.txt" )
        with open(out_file_path, "w") as output_file:
            count = -1
            for line in tqdm.tqdm(input_file):
                count += 1
                if count < sequence_start_idx:
                    continue

                # Load the TUM-formatted estimates
                values = list(map(float,line[:-1].split(" ")))
                timestamp = values[0]
                x,y,z = values[1:4]
                qx, qy, qz, qw = values[4:]
                R = Rotation([qx,qy,qz,qw], normalize=True).as_matrix()
                
                # Fill transformation matrix of the estimations
                T_from_imu_to_utm = np.eye(4,4)
                T_from_imu_to_utm[:3,:3] = R
                T_from_imu_to_utm[:3,3] = [x,y,z]

                # To local frame
                if T_from_utm_to_imu_init is None:
                     T_from_utm_to_imu_init = np.linalg.inv(T_from_imu_to_utm)
                T_from_imu_curr_to_imu_init = T_from_utm_to_imu_init @ T_from_imu_to_utm

                # Transform to the camera frame
                T_from_cam0_curr_to_cam0_init = T_from_imu_to_cam0 @ T_from_imu_curr_to_imu_init @ T_from_cam0_to_imu

                # Output lines
                out_content = " ".join( map(str, T_from_cam0_curr_to_cam0_init[:3,:].flatten()) )
                out_line = f"{out_content}\n"
                output_file.write(out_line)

                if count >= sequence_end_idx:
                    break
    exit(0)