import numpy as np
from scipy.spatial.transform import Rotation

def kitti_odometry_line_to_transform(input_line : str) -> np.ndarray :
    # Load line elements
    input_tokens = input_line[:-1].split(" ")
    input_array = np.array( list(map(float, input_tokens)) )

    # Get input matrix
    T_input = input_array.reshape(3,4)

    # Fill output matrix
    T_output = np.eye(4,4)
    T_output[:3,:4] = T_input
    
    return T_output