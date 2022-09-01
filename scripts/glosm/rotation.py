import numpy as np

def normalize_orientation(orientation_array : np.ndarray) -> np.ndarray:
    """Normalizes the orientation values so that they range between [-pi,pi].
    """
    normalized = orientation_array
    normalized[ normalized >  np.pi ] -= 2 * np.pi
    normalized[ normalized < -np.pi ] += 2 * np.pi
    return normalized.copy()

def build_2d_rotation_matrix(yaw : float) -> np.ndarray:
    return np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])

def normalize_angle(angle : float) -> float:
    out_angle = angle
    while out_angle > np.pi: out_angle -= 2 * np.pi
    while out_angle <-np.pi: out_angle += 2 * np.pi
    return out_angle

def build_rotation_matrix_around_axis(angle : float, axis : str) -> np.ndarray:
    """Builds the rotation matrix around the given axis an angle (in radians)."""
    assert axis in ["x", "y", "z"], "Provided axis not contained among the options."
    c = np.cos(angle)
    s = np.sin(angle)
    R = None
    if axis == "x":
        R = np.array([
            [1., 0., 0.],
            [0., c, -s ],
            [0., s,  c],
        ])
    elif axis == "y":
        R = np.array([
            [ c , 0 , s ],
            [ 0 , 1 , 0 ],
            [-s , 0 , c ]
        ])
    elif axis == "z":
        R = np.array([
            [ c , -s , 0 ],
            [ s ,  c , 0 ],
            [ 0 ,  0 , 1 ]
        ])
    return R

def rotation_matrix_from_euler_angles(roll : float, pitch : float, yaw : float) -> np.ndarray:
    """Builds the rotation matrix given the Euler angles (in radians). 
    
    The rotation matrix performs first the rotation around the 'x' axis, after 
    around the 'y' axis and last around the 'z' axis ('xyz' convention).
    """
    c = np.cos
    s = np.sin
    R_00 = c(yaw) * c(pitch)
    R_01 = c(yaw) * s(pitch) * s(roll) - s(yaw) * c(roll)
    R_02 = c(yaw) * s(pitch) * c(roll) + s(yaw) * s(roll)
    R_10 = s(yaw) * c(pitch)
    R_11 = s(yaw) * s(pitch) * s(roll) + c(yaw) * c(roll)
    R_12 = s(yaw) * s(pitch) * c(roll) - c(yaw) * s(roll)
    R_20 = -s(pitch)
    R_21 = c(pitch) * s(roll)
    R_22 = c(pitch) * c(roll)
    R = np.array([
        [ R_00, R_01, R_02 ],
        [ R_10, R_11, R_12 ],
        [ R_20, R_21, R_22 ],
    ])
    return R

def dR_dphi(phi : float, theta : float, psi : float) -> np.ndarray:
    c = np.cos
    s = np.sin

    dR = np.array([
        [0., c(psi) * s(theta) * c(phi) + s(psi) * s(phi)   , -c(psi) * s(theta) * s(phi) + s(psi) * c(phi)  ],
        [0., s(psi) * s(theta) * c(phi) - c(psi) * s(phi)   , -s(psi) * s(theta) * s(phi) - c(psi) * c(phi) ],
        [0., c(theta) * c(phi)                              , -c(theta) * s(phi) ]
    ])

    return dR

def dR_dtheta(phi : float, theta : float, psi : float) -> np.ndarray:
    c = np.cos
    s = np.sin

    dR = np.array([
        [ -c(psi) * s(theta), c(psi) * c(theta) * s(phi), c(psi) * c(theta) * c(phi) ],
        [ -s(psi) * s(theta), s(psi) * c(theta) * s(phi), s(psi) * c(theta) * c(phi) ],
        [ -c(theta)         , -s(theta) * s(phi)        , -s(theta) * c(phi)         ]
    ])

    return dR

def dR_dpsi(phi : float, theta : float, psi : float) -> np.ndarray:
    c = np.cos
    s = np.sin

    dR = np.array([
        [-s(psi) * c(theta) , -s(psi) * s(theta) * s(phi) - c(psi) * c(phi) , -s(psi) * s(theta) * c(phi) + c(psi) * s(phi) ],
        [ c(psi) * c(theta) ,  c(psi) * s(theta) * s(phi) - s(psi) * c(phi) ,  c(psi) * s(theta) * c(phi) + s(psi) * s(phi) ],
        [ 0                 ,  0.                                           ,  0.  ]
    ])
    return dR

def partial_rotation_matrix(roll : float, pitch : float, yaw : float, partial : str) -> np.ndarray:
    """Computes the partial derivative of the rotation matrix w.r.t. the given variable.
    
    Assumes the 'xyz' convention.

    Args
    -------
    roll,pitch,yaw: The Euler angles in radians.
    partial: The variable for computing the partial (accepted values: "roll", 
        "pitch" or "yaw").

    Returns
    -------
    The partial derivative of the rotation matrix.
    """
    assert partial in ["roll", "pitch", "yaw"], "Provided `partial` argument not implemented."
    s = np.sin
    c = np.cos
    partial_R = None

    if partial == "roll":
        partial_R = dR_dphi(roll, pitch, yaw)

    elif partial == "pitch":
        partial_R = dR_dtheta(roll, pitch, yaw)
        
    elif partial == "yaw":
        partial_R = dR_dpsi(roll, pitch, yaw)

    return partial_R