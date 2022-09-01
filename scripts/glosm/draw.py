
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import Polygon, Point

def draw_pose_2d(x : float, y : float, yaw : float, ax : plt.Axes, label : Optional[str] = None) -> None:
    ax.scatter([x], [y], marker="x", color="black")

    # Draw coordinates system
    dx = 1.5 * np.cos(yaw)
    dy = 1.5 * np.sin(yaw)
    ax.arrow( x, y, dx, dy, color="red" )
    dx = 1.5 * np.cos(yaw + (np.pi / 2) )
    dy = 1.5 * np.sin(yaw + (np.pi / 2))
    ax.arrow( x, y, dx, dy, color="green" )
    
    if label is not None:
        ax.text( x,y, s=label)

def draw_navigable_area(driveable_map : gpd.GeoDataFrame, ax : plt.Axes) -> None:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        bounding_box_poly = Polygon([
            Point(x_min,y_min), # Bottom left
            Point(x_max,y_min), # Bottom right
            Point(x_max,y_max), # top right
            Point(x_min,y_max), # top left
        ])
        bounding_box_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(bounding_box_poly, crs = driveable_map.crs) )
        visible_driveable_map = driveable_map.overlay(bounding_box_gdf, keep_geom_type=True)
        gpd.GeoSeries(visible_driveable_map.geometry.unary_union, crs=driveable_map.crs).plot(ax=ax, alpha=0.7, color="gray")

def draw_traffic_signals(traffic_signals_map : gpd.GeoDataFrame , ax : plt.Axes) -> None:
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    bounding_box_poly = Polygon([
        Point(x_min,y_min), # Bottom left
        Point(x_max,y_min), # Bottom right
        Point(x_max,y_max), # top right
        Point(x_min,y_max), # top left
    ])
    bounding_box_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(bounding_box_poly, crs = traffic_signals_map.crs) )
    visible_traffic_signals_map = traffic_signals_map.overlay(bounding_box_gdf, keep_geom_type=True)
    visible_traffic_signals_map.buffer(0.8).geometry.plot(ax=ax, alpha=1.0, color="red")
    for _, traffic_signal_row in visible_traffic_signals_map.iterrows():
        easting = traffic_signal_row.geometry.x
        northing = traffic_signal_row.geometry.y
        ax.text(easting, northing, s="Traffic Signal")