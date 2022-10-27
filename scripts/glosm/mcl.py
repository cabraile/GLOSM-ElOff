from typing import Any, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

import geopandas as gpd
from shapely.geometry import Point, Polygon
import rasterio
import rasterio.transform

from scipy.stats import circmean

from glosm.rotation import normalize_orientation, build_2d_rotation_matrix
from glosm.draw import draw_pose_2d

def low_variance_sampler(weights : np.ndarray) -> np.ndarray:
  """Samples ids proportional to the weights provided.

  Parameters
  ------------
  weights:
      The weights of the particles. Will be normalized in this function.

  Returns
  ------------
  The new ids for sampling the particles with replacement proportional to their 
  weights.
  """
  sum_w = np.sum(weights)
  if(sum_w == 0):
      return np.arange(0, weights.size)
  w = weights / sum_w
  n_particles = w.size
  delta = 1./n_particles
  r_init = np.random.rand() * delta
  ids = np.zeros((n_particles),dtype=int)
  
  i = 0
  cumulative_w = w[0]
  for k in range(n_particles):
    # The next cumulative weight has to be greater than this
    r = r_init + k * delta
    while r > cumulative_w:
      # Increment the cumulative weight: still not enough
      i += 1
      cumulative_w += w[i]
    ids[k] = i
    
  return ids

def empirical_mean_and_covariance(poses : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Estimate the mean and covariance of the normal distribution that best fits 
  the data.
  """
  mean_position = np.average(poses[:,:2],axis=0).flatten()
  mean_orientation = scipy.stats.circmean(poses[:,2], low=-np.pi, high=np.pi)
  position_covariance = np.cov(poses[:,:2].T)
  orientation_variance = scipy.stats.circvar(poses[:,2])
  mean = np.hstack([mean_position, mean_orientation])
  covariance = np.zeros((3,3))
  covariance[:2,:2] = position_covariance
  covariance[2,2] = orientation_variance
  return mean, covariance

class MCLXYZYaw:
    """Monte Carlo Localization."""

    def __init__(self) -> None:
        self.particles              = None

        # Map layers
        self.dsm_array              = None
        self.dsm_transform          = None
        self.stop_signs_map         = None
        self.local_stop_signs_map   = None
        self.driveable_map          = None
        self.local_driveable_map    = None
        self.traffic_signals_map    = None
        self.local_traffic_signals_map = None

        # Local boundaries
        self.local_map_boundaries   = None
        self.local_map_size         = 200.0
    
    # Map-related
    # ================================

    def should_update_local_map(self, tolerance_meters : float) -> bool:
        x_min, y_min = np.min(self.particles[:,:2], axis=0)
        x_max, y_max = np.max(self.particles[:,:2], axis=0)
        is_x_boundary_ok = ( self.local_map_boundaries["x_min"] <= x_min - tolerance_meters ) and ( self.local_map_boundaries["x_max"] >= x_max + tolerance_meters ) 
        is_y_boundary_ok = ( self.local_map_boundaries["y_min"] <= y_min - tolerance_meters ) and ( self.local_map_boundaries["y_max"] >= y_max + tolerance_meters ) 
        return not ( is_x_boundary_ok and is_y_boundary_ok )

    def set_digital_surface_model_map(self, raster : rasterio.DatasetReader) -> None:
        self.dsm_array = raster.read().squeeze()
        self.dsm_transform = raster.transform

    def set_driveable_map(self, driveable_map : gpd.GeoDataFrame) -> None:
        self.driveable_map = driveable_map
        self.local_driveable_map = None

    def set_traffic_signals_map(self, traffic_signals_map : gpd.GeoDataFrame) -> None:
        self.traffic_signals_map = traffic_signals_map
        self.local_traffic_signals_map = None

    def set_stop_signs_map(self, stop_signs_map : gpd.GeoDataFrame) -> None:
        self.stop_signs_map = stop_signs_map
        self.local_stop_signs_map = None

    def load_local_region_map(self, center_x : float, center_y : float, region_size : float ) -> None:
        # Create bounding box area
        x_min = center_x - region_size/2.0
        x_max = center_x + region_size/2.0
        y_min = center_y - region_size/2.0
        y_max = center_y + region_size/2.0
        self.local_map_boundaries = {
            "x_min" : x_min, "x_max" : x_max, "y_min" : y_min, "y_max" : y_max
        }
        bounding_box_poly = Polygon([
            Point(x_min,y_min), # Bottom left
            Point(x_max,y_min), # Bottom right
            Point(x_max,y_max), # top right
            Point(x_min,y_max), # top left
        ])
        bounding_box_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(bounding_box_poly, crs = self.driveable_map.crs) )
        
        # Filter local elements
        self.local_driveable_map = self.driveable_map.overlay(bounding_box_gdf, keep_geom_type=True)
        self.local_traffic_signals_map = self.traffic_signals_map.overlay(bounding_box_gdf, keep_geom_type=True)
        self.local_stop_signs_map = self.stop_signs_map.overlay(bounding_box_gdf, keep_geom_type=True)

    # ================================

    def get_particles(self) -> np.ndarray:
        return self.particles

    def get_dsm_z_at(self, xy_array : np.ndarray) -> np.ndarray:
        """    
        """
        # Retrieve the row and column for each particle
        particles_xy = xy_array[:,:2]
        particles_row_col = rasterio.transform.rowcol(transform=self.dsm_transform, xs=particles_xy[:,0], ys=particles_xy[:,1])
        particles_row_col_array = np.stack(particles_row_col, axis=1)
        
        # Get the mask of particles that are inside the DSM
        nrows, ncols = self.dsm_array.shape
        particles_rows = np.array(particles_row_col[0])
        particles_cols = np.array(particles_row_col[1])
        mask_oob = (particles_rows < 0) | (particles_rows >= nrows) | (particles_cols < 0) | (particles_cols >= ncols)
        mask_in_bounds = ~mask_oob

        # Get the Z for each particle in bounds.
        dsm_z = np.full((len(self.particles)), 0.0)
        dsm_z[mask_in_bounds] = self.dsm_array[particles_row_col_array[mask_in_bounds,0], particles_row_col_array[mask_in_bounds,1]]
        
        # Particles that are inside nan positions (not filled by the DSM) will not have their offsets changed
        dsm_z[np.isnan(dsm_z)] = 0.0
        return dsm_z

    def plot(self, ax : plt.Axes, draw_poses : bool = False) -> None:
        if draw_poses:
            for i in range(len(self.particles)):
                x,y, yaw = self.particles[i]
                draw_pose_2d(x,y,yaw,ax, arrow_size=1.0)
        else:
            ax.scatter(self.particles[:,0], self.particles[:,1], c="orange", marker="x")
        
    def get_mean(self) -> np.ndarray:
        """Returns the (x,y,z,yaw) array of the particles' mean."""
        x_mean, y_mean, z_mean = np.average(self.particles[:,:3], axis=0).flatten()
        yaw_mean = circmean(self.particles[:,3], high=np.pi, low = -np.pi,)
        return np.array([ x_mean, y_mean, z_mean, yaw_mean ])

    def get_covariance(self) -> np.ndarray:
        """Returns the covariance array of the variables x,y, z and yaw."""
        # TODO: the covariance returned does not consider the circular nature 
        # of the yaw
        covariance = np.cov(self.particles.T)
        return covariance

    # Data flow
    # =========================================================================

    def sample(self, mean : np.ndarray, covariance : np.ndarray, n_particles : int) -> None:
        """Sample points around the provided mean."""
        assert (mean.size == 4) and (covariance.shape[0] == 4), "Shapes should be 3-dimensional"
        self.particles = np.random.multivariate_normal(mean = mean, cov = covariance, size = n_particles)
        self.particles[:,3] = normalize_orientation(self.particles[:,3])

    def predict(
        self, 
        delta_xyz : Tuple[float], 
        delta_yaw : float,
        position_covariance : np.ndarray,
        orientation_variance : float
    ) -> None:
        """Performs state prediction based on the control commands given and the 
        covariance matrix.

        The control array should be provided as an

        """
        # Project the control to each of the particles' coordinates
        offsets_list = []
        translation = np.array(delta_xyz).reshape(3,1)
        for i in range(len(self.particles)):
            rotation_matrix = build_2d_rotation_matrix(self.particles[i,3])
            offset_x, offset_y = rotation_matrix @ translation[:2]
            offset_z = translation[2]
            offsets_list.append( (offset_x, offset_y, offset_z ) )
        
        offsets_array = np.vstack(offsets_list).reshape(-1,3)
        broadcast_yaw = np.array([delta_yaw] * len(self.particles)).reshape(-1,1)
        u_proj = np.hstack([ offsets_array, broadcast_yaw ]).reshape(-1,4)

        # Sum the noisy control for each particle's state
        cov = np.eye(4)
        cov[:3,:3] = position_covariance
        cov[3,3] = orientation_variance
        noise_array = np.random.multivariate_normal( mean=np.zeros((4,)), cov=cov , size=len(self.particles))
        new_particles = self.particles + ( u_proj + noise_array.reshape(-1,4) )
        new_particles[:,3] = normalize_orientation(new_particles[:,3])

        # Replace old particle values by the new ones
        self.particles = new_particles

    def resample(self, weights : np.ndarray) -> None:
        ids = low_variance_sampler(weights)
        self.particles = self.particles[ids]

    # =========================================================================

    # Update-related
    # =========================================================================

    def weigh_in_driveable_area(self, prob_inside_navigable_area : float = 0.90) -> None:
        # First check local map
        if self.should_update_local_map(tolerance_meters=15.0):
            x_min, y_min = np.min(self.particles[:,:2], axis=0)
            x_max, y_max = np.max(self.particles[:,:2], axis=0)
            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            local_map_size = max(x_max - x_min, y_max - y_min)  + self.local_map_size
            self.load_local_region_map(center_x, center_y, local_map_size)

        particles_as_points = [ Point(self.particles[i,0], self.particles[i,1]) for i in range(len(self.particles)) ]
        weights = []
        for point in particles_as_points:
            is_inside_any = np.any(self.local_driveable_map.geometry.contains(point))
            if is_inside_any:
                w = prob_inside_navigable_area
            else:
                w = 1 - prob_inside_navigable_area
            weights.append(w)
        weights = np.array( weights )
        self.resample(weights)

    def weigh_traffic_signal_detection(self, sensitivity : float, false_positive_rate : float) -> None:
        particles_as_points = [ Point(self.particles[i,0], self.particles[i,1]) for i in range(len(self.particles)) ]
        weights = []
        normalizer_factor = 1. / (sensitivity + false_positive_rate)
        for particle_idx, point in enumerate(particles_as_points):
            # Query the closest traffic signal registered to the particle
            closest_traffic_signal_index = self.traffic_signals_map.sindex.nearest(point)[1,0]
            traffic_signal_point = self.traffic_signals_map.iloc[closest_traffic_signal_index].geometry

            # Compute distance and orientation between the traffic signal and the particle's frame
            particle_array = np.array([point.x, point.y])
            traffic_signal_array = np.array( [ traffic_signal_point.x, traffic_signal_point.y ] )

            # Check if the traffic signal should be visible to the particle
            particle_orientation = self.particles[particle_idx,3]
            R_from_particle_to_world = np.array([
                [np.cos(particle_orientation), - np.sin(particle_orientation)],
                [np.sin(particle_orientation), np.sin(particle_orientation)]
            ])

            # Project traffic signal to particle frame
            T_from_particle_to_world = np.eye(3)
            T_from_particle_to_world[:2,:2] = R_from_particle_to_world
            T_from_particle_to_world[:2, 2] = particle_array.flatten()
            T_from_world_to_particle = np.linalg.inv(T_from_particle_to_world)
            traffic_signal_array_homog = np.vstack( [ traffic_signal_array.reshape(2,1), np.array([1.0]).reshape(1,1) ] )
            traffic_signal_particle_frame = (T_from_world_to_particle @ traffic_signal_array_homog).flatten()[:2]

            # Check if should be visualized
            bearing_magnitude = np.linalg.norm(traffic_signal_particle_frame)
            bearing_angle_rad = np.arctan2(traffic_signal_particle_frame[1], traffic_signal_particle_frame[0])
            bearing_angle_deg = np.rad2deg(bearing_angle_rad)
            is_in_sight = ( bearing_magnitude < 50.0 ) and ( np.abs(bearing_angle_deg) < 55.0 )
            if is_in_sight:
                weights.append( normalizer_factor * sensitivity)
            else:
                weights.append( normalizer_factor * false_positive_rate)
        
        weights = np.array( weights )
        self.resample(weights)

    def weigh_eloff(self, dsm_z_std : float = 0.1) -> None:
        # Compute the weights
        dsm_z = self.get_dsm_z_at(self.particles[:,:2])
        particles_z = self.particles[:,2]
        mahalanobis = np.abs( particles_z - dsm_z)/dsm_z_std
        weights = 1./(mahalanobis+0.001)

        # If the sum of weights is pretty low, do not resample
        sum_w = np.sum(weights)
        if sum_w <= 1e-10:
            return
        
        # Resample
        weights_norm = weights / np.sum(sum_w)
        self.resample(weights_norm)

    # =========================================================================