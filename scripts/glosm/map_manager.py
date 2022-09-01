import os
import pyrosm #https://pyrosm.readthedocs.io/en/latest/basics.html
from scipy.spatial.transform import Rotation
from pyproj import CRS
import geopandas as gpd
import utm
from typing import Dict, Tuple

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

def load_map_layers(osm_dir : str) -> Dict[str,gpd.GeoDataFrame]:
    """
    Returns
    ------------
    A dictionary containing the following keys:
        "driveable_area":
            The driveable area polygons' GeoDataFrame.
        "traffic_signals":
            The traffic signals' points indicated on the map as a GeoDataFrame.
    """
    road_network_path = os.path.join(osm_dir,f"road_network_utm.shp")
    traffic_signals_path = os.path.join(osm_dir,f"traffic_signals_utm.shp")
    stop_signs_path = os.path.join(osm_dir,f"stop_signs_utm.shp")

    # Loads the base map if any of the necessary elements were cached before
    if ( 
        (not os.path.exists(road_network_path)) or 
        (not os.path.exists(traffic_signals_path)) or 
        (not os.path.exists(stop_signs_path))
    ):
        print("> Loading .osm.pbm")
        map_file_path = pyrosm.get_data("karlsruhe", directory=os.path.abspath(osm_dir))
        osm_map = pyrosm.OSM(map_file_path)
    
    # Loads the driveable area
    if not os.path.exists(road_network_path):
        print("> Filtering road network elements")
        drive_network = osm_map.get_network(network_type="driving")
        drive_network, map_utm_crs = project_geodataframe_to_best_utm(drive_network)
        print("> Buffering navigable area - this might take long (~5 minutes)")
        # Assign one lane for lanes that were undefined
        drive_network["lanes"].fillna(value="1",inplace=True)
        # Fix wrong lane numbers
        drive_network["lanes"] = drive_network["lanes"].apply(lambda x : x.replace(",","."))
        drive_network["lanes"] = drive_network["lanes"].astype(float)
        # Buffer the area for each lane
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

    # Loads the traffic signs
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
        print(f"> Saved projected traffic signals cache to {traffic_signals_path}")
    
    # Read from cache
    else:
        print(f"> Traffic signs cached file found at {traffic_signals_path}")
        traffic_signals = gpd.read_file(traffic_signals_path)
        map_utm_crs = drive_network.crs
    
    # Loads the stop signs
    if not os.path.exists(stop_signs_path):
        print(f"> Filtering stop signs")
        stop_signs = osm_map.get_data_by_custom_criteria(
            custom_filter={"highway":["stop"]}, # Keep data matching the criteria above
            filter_type="keep",
            # Do not keep nodes (point data)    
            keep_nodes=True, 
            keep_ways=False, 
            keep_relations=False
        )
        stop_signs,_ = project_geodataframe_to_best_utm(stop_signs)
        stop_signs.to_file(stop_signs_path)
        print(f"> Saved the projected stop signs to {stop_signs_path}")
    
    # Load from cache
    else:
        print(f"> Stop signs cached file found at {stop_signs_path}")
        stop_signs = gpd.read_file(stop_signs_path)
        map_utm_crs = drive_network.crs

    # Return the dictionary of layers
    layers = {
        "driveable_area"    : drive_network,
        "traffic_signals"   : traffic_signals,
        "stop_signs"        : stop_signs
    }
    return layers
