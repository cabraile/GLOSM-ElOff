import os
import sys
import argparse
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import contextily as cx

# Importing GLOSM
MODULES_PATH =os.path.realpath( 
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), # root/scripts/demos/
        os.path.pardir  # root/scripts/
    )
)
sys.path.append(MODULES_PATH)
from glosm.map_manager import get_utm_crs_from_lat_lon

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True, help="Path list to the CSV of the estimated trajectories.", nargs="+")
    parser.add_argument("--reference", required=True, help="File name (without extension) of the reference.")
    parser.add_argument("--latitude", required=True, type=float, help="A reference latitude for loading with the correct CRS.")
    parser.add_argument("--longitude", required=True, type=float, help="A reference longitude for loading with the correct CRS.")
    args = parser.parse_args()

    # Load reference CRS
    crs = get_utm_crs_from_lat_lon(args.latitude, args.longitude)

    # Load CSV files
    print("> Loading trajectories' CSV files")
    trajectories = {}
    for trajectory_csv_path in tqdm.tqdm(args.trajectories):
        # Get trajectory name
        trajectory_csv_path = os.path.abspath(trajectory_csv_path)
        trajectory_csv_basename = os.path.basename(trajectory_csv_path)
        trajectory_name = os.path.splitext(trajectory_csv_basename)[0]

        # Load CSV as DataFrame and assign
        trajectory_df = pd.read_csv(trajectory_csv_path, sep=",", header=0)
        trajectories[trajectory_name] = trajectory_df

    # Generate random colors for each trajectory
    trajectories_colors = {}
    for trajectory_name, trajectory_df in tqdm.tqdm(trajectories.items()):
        trajectories_colors[trajectory_name] = tuple(np.random.rand(3))

    # Draw trajectories
    print("> Drawing the trajectories")
    fig, ax = plt.subplots(1,1,figsize=(15,15))
    for trajectory_name, trajectory_df in tqdm.tqdm(trajectories.items()):
        trajectory_df.plot(ax=ax, x="x", y="y", color=trajectories_colors[trajectory_name], label=trajectory_name, linewidth=7.0)
    cx.add_basemap(ax, crs=crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)
    plt.savefig(f"results/trajectories.png")

    # Plot error lines
    print("> Plotting error lines")
    ax.cla()
    for trajectory_name, estimation_df in tqdm.tqdm(trajectories.items()):
        if trajectory_name == args.reference:
            continue
        reference_df = trajectories[args.reference]
        error_list = []

        for _, estimation_row in estimation_df.iterrows():
            # Get the closest groundtruth estimation in time
            time_deltas = np.abs( reference_df["elapsed_time"] - estimation_row["elapsed_time"] )
            closest_reference_idx = np.argmin(time_deltas)
            reference_row = reference_df.iloc[closest_reference_idx]

            # Compute euclidean distance between reference and estimation
            estimation_xy = np.array( estimation_row[["x","y"]] )
            reference_xy = np.array( reference_row[["x", "y"]] )
            distance = np.linalg.norm( estimation_xy - reference_xy )
            error_list.append(distance)

        # Assign computed errors to trajectory's dataframe
        estimation_df["error"] = error_list

        # Plot
        estimation_df.plot(x="elapsed_time",y="error", label=trajectory_name, color=trajectories_colors[trajectory_name], ax=ax)

    plt.savefig(f"results/error.png")

    # Output statistics
    summary_dict = {}
    for trajectory_name, estimation_df in trajectories.items():
        if trajectory_name == args.reference:
            continue
        summary_series = estimation_df["error"].describe()
        summary_dict[trajectory_name] = summary_series
        print(f"Trajectory '{trajectory_name}' statistics")
        print(f"-----------------------------------------")
        print(summary_series)
        print()
    summary_df = pd.DataFrame(summary_dict).T
    summary_df.to_csv(f"results/summary.csv",index=True)
    return 0

if __name__ == "__main__" : 
    sys.exit(main())