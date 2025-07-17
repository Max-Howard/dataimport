import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import contextily as ctx
import geopandas as gpd
from import_ptf import read_ptf

FT_TO_M = 0.3048
TIME_GAP = 60
ALT_GAP_MAX = 100 # Maximum altitude gap in meters before interpolation
CRUISE_STEP = 50 * 1852       # 50 nautical miles in meters
DELTA_ALT_CRUISE = 1000 * FT_TO_M
AIRPORT_DATABASE = pd.read_csv("airports.csv").set_index("icao")
FLIGHTS_DIRECTORY = "FlightData/Processed"
BADA_AC_CRUISE_LEVELS = pd.read_csv("cruise_fl.csv").set_index("typecode")


def load_flight_paths(origin_filter=None, destination_filter=None, typecode_filter=None, limit=None) -> dict[pd.DataFrame]:
    """
    Loads flight path data from CSV files in a specified directory, applying optional filters for origin, destination, and aircraft type code.
    Args:
        origin_filter (Iterable[str], optional): List or set of allowed origin airport codes. If None, no filter is applied.
        destination_filter (Iterable[str], optional): List or set of allowed destination airport codes. If None, no filter is applied.
        typecode_filter (Iterable[str], optional): List or set of allowed aircraft type codes. If None, no filter is applied.
        limit (int, optional): Maximum number of flight paths to load. If None, all matching files are loaded.
    Returns:
        dict[str, pd.DataFrame]: A dictionary mapping flight identifiers (derived from filenames) to their corresponding DataFrames.
    Notes:
        - Expects CSV filenames in the format: ORIGIN_DESTINATION_TYPECODE_ICAO24_FLIGHTNUMBER.csv
        - Each DataFrame contains the flight path data for a single flight.
    """

    flight_paths = {}
    for file in os.listdir(FLIGHTS_DIRECTORY):
        if not file.endswith('.csv') or not file.endswith('.csv'):
            continue
        origin, destination, typecode, icao24, flight_number = file.strip(".csv").split("_")

        if origin_filter and origin not in origin_filter:
            continue
        if destination_filter and destination not in destination_filter:
                continue
        if typecode_filter and typecode not in typecode_filter:
            continue
        if limit and len(flight_paths) >= limit:
                break

        file_path = os.path.join(FLIGHTS_DIRECTORY, file)
        df = pd.read_csv(file_path)

        flight_paths[file.replace('.csv', '')] = df
    print(f"Loaded {len(flight_paths)} flight paths.")
    return flight_paths

def find_ptf_data_from_alt(alt, ac_type):
            ptf_filepath = f"BADA/{ac_type}__.ptf"
            ptf = read_ptf(ptf_filepath)
            ac_data = ptf['table']
            closest_FL = ac_data.index[np.abs(ac_data.index - alt / 100 / FT_TO_M).argmin()]
            return ac_data.loc[closest_FL]


def plot_overall(flight_paths, color_by="gs"):
      """
      Plot flight paths on a map with optional color coding.
      color_by: "gs", "tas", "geoaltitude", "baroaltitude", or "index"
      """
      plt.figure(figsize=(12, 6))
      ax = plt.axes(projection=ccrs.PlateCarree())
      ax.add_feature(cfeature.LAND)
      ax.add_feature(cfeature.OCEAN)
      ax.add_feature(cfeature.COASTLINE)
      ax.add_feature(cfeature.BORDERS, linestyle=':')
      ax.add_feature(cfeature.LAKES, alpha=0.5)
      ax.add_feature(cfeature.RIVERS)

      min_lon = min(min(flight_path['lon']) for flight_path in flight_paths.values())
      max_lon = max(max(flight_path['lon']) for flight_path in flight_paths.values())
      min_lat = min(min(flight_path['lat']) for flight_path in flight_paths.values())
      max_lat = max(max(flight_path['lat']) for flight_path in flight_paths.values())
      lon_margin = (max_lon - min_lon) * 0.1
      lat_margin = (max_lat - min_lat) * 0.1
      min_lon -= lon_margin
      max_lon += lon_margin
      min_lat -= lat_margin
      max_lat += lat_margin

      ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

      if color_by == "gs":
            cbar_label = "Ground Speed (m/s)"
            cbar_source = "gs"
            max_val = max(max(flight_path[cbar_source]) for flight_path in flight_paths.values())
            min_val = min(min(flight_path[cbar_source]) for flight_path in flight_paths.values())
            norm = Normalize(vmin=min_val, vmax=max_val)
      elif color_by == "tas":
            cbar_label = "Air Speed (m/s)"
            cbar_source = "tas"
            max_val = max(max(flight_path[cbar_source]) for flight_path in flight_paths.values())
            min_val = min(min(flight_path[cbar_source]) for flight_path in flight_paths.values())
            norm = Normalize(vmin=min_val, vmax=max_val)
      elif color_by.endswith("altitude"):
            if color_by == "geoaltitude":
                  cbar_label = "GNSS Altitude (m)"
                  cbar_source = "geoaltitude"
            elif color_by == "baroaltitude":
                  cbar_label = "Barometric Altitude (m)"
                  cbar_source = "baroaltitude"
            max_val = min(5000, max(max(flight_path[cbar_source]) for flight_path in flight_paths.values()))
            min_val = min(min(flight_path[cbar_source]) for flight_path in flight_paths.values())
            norm = Normalize(vmin=min_val, vmax=max_val)
      elif color_by == "index":
            cbar_label = "Index"
            cbar_source = None
            max_val = max(len(flight_path) for flight_path in flight_paths.values())
            min_val = 0
            norm = Normalize(vmin=min_val, vmax=max_val)
      else:
            raise ValueError(f"Unsupported color_by: {color_by}")

      # Plot the flight paths
      for flight_path_name, flight_path in flight_paths.items():
            # Remove the interpolated points before plotting the scatter points
            if 'interpolated' in flight_path.columns:
                  plt.plot(flight_path['lon'], flight_path['lat'], c='red', linestyle='-', linewidth=0.5, alpha=0.7)
                  flight_path = flight_path[~flight_path['interpolated']]

            if color_by == "index":
                  c = np.arange(len(flight_path))
            else:
                  c = flight_path[cbar_source]
            points = plt.scatter(flight_path['lon'], flight_path['lat'], c=c, cmap='viridis_r', marker=".", norm=norm, s=8, alpha=1.0)

      cbar = plt.colorbar(points, orientation='horizontal', pad=0.05, aspect=50, label=cbar_label)
      cbar.ax.set_position([0.215, -0.05, 0.6, 0.3])  # Adjust the position and size of the colorbar
      plt.gca().xaxis.set_major_locator(plt.NullLocator())
      plt.gca().yaxis.set_major_locator(plt.NullLocator())
      plt.legend()
      plt.show()


def detail_plot(flight_paths):
      """
      Plot flight data points on a map.
      Used to visualize coverage at the beginning/end of flights.
      """

      # Define the center as the final lat and lon value of the first flight path
      first_flight_path = next(iter(flight_paths.values()))
      cen_lat = first_flight_path['lat'].iloc[-1]
      cen_lon = first_flight_path['lon'].iloc[-1]

      # Define the size of the plot area
      plot_size_nm = 2.5
      lat_step = plot_size_nm * 1.852 / 110.574
      lon_step = plot_size_nm * 1.852 / (111.320 * np.cos(np.radians(cen_lat)))

      # Filter points within the plot area
      lat = []
      lon = []
      velocities = []
      altitudes = []
      for flight_path_name, flight_path in flight_paths.items():
            mask = (
                  (flight_path['lat'] >= cen_lat - lat_step) &
                  (flight_path['lat'] <= cen_lat + lat_step) &
                  (flight_path['lon'] >= cen_lon - lon_step) &
                  (flight_path['lon'] <= cen_lon + lon_step) &
                  (flight_path['geoaltitude'] <= 750)
            )
            lat.extend(flight_path['lat'][mask])
            lon.extend(flight_path['lon'][mask])
            velocities.extend(flight_path['tas'][mask])
            altitudes.extend(flight_path['geoaltitude'][mask])

      # Create a GeoDataFrame for the scatter points
      gdf_flight_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")
      gdf_flight_points = gdf_flight_points.to_crs(epsg=3857)

      # Plot the scatter points
      fig, ax = plt.subplots(figsize=(6, 6))
      scatter = ax.scatter(gdf_flight_points.geometry.x, gdf_flight_points.geometry.y, c=altitudes, cmap='plasma', marker='.', s=5, alpha=0.7, vmin=0, vmax=750)
      cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal',fraction=0.046, pad=0.04) #, pad=0.01, aspect=50
      cbar.set_label('Altitude (m)')


      ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=14)
      ax.set_aspect('equal', adjustable='box')
      ax.set_axis_off()
      plt.show()

def baro_vs_geo_altitude_plot(flight_paths):
      plt.figure(figsize=(10, 5))
      for flight_path_name, flight_path in flight_paths.items():
            time_series = (flight_path['time'] - flight_path["time"][0]) / 60**2
            plt.plot(time_series, flight_path['geoaltitude'], color="orange", label=f"{flight_path_name} Geo Altitude")
            plt.scatter(time_series, flight_path['geoaltitude'], color="orange", s=5)
            plt.plot(time_series, flight_path['baroaltitude'], color="blue", label=f"{flight_path_name} Baro Altitude")
            plt.scatter(time_series, flight_path['baroaltitude'], color="blue", s=5)
      plt.xlabel("Time (hours)")
      plt.ylabel("Altitude (m)")
      plt.show()


def compare_with_bada(flight_paths):
      """
      Compare flight altitudes with BADA predicted altitudes.
      Plots BADA altitudes for each flight path along with the actual barometric altitude.
      """
      plt.figure(figsize=(10, 5))
      colors = plt.colormaps.get_cmap('tab10').resampled(len(flight_paths))
      for i, (flight_path_name, flight_path) in enumerate(flight_paths.items()):
            bada_timestep = 1 # seconds
            bada_times = np.arange(0, flight_path["time"].iloc[-1]-flight_path["time"][0], bada_timestep) / 60**2
            origin_icao, destination_icao, ac_type, aircraft_icao, flight_id = flight_path_name.split("_")
            ptf_filepath = f"BADA/{ac_type}__.ptf"
            ptf = read_ptf(ptf_filepath)
            ac_data = ptf['table']
            alt_start = AIRPORT_DATABASE.loc[origin_icao, "alt"] * FT_TO_M
            alt_stop = AIRPORT_DATABASE.loc[destination_icao, "alt"] * FT_TO_M
            alt_cruise = BADA_AC_CRUISE_LEVELS.loc[ac_type, "cr_fl"] * 100 * FT_TO_M

            while True: # Loop to adjust cruise altitude if necessary (may not have time to reach cruise altitude)

                  # Climb from start altitude to cruise altitude (climb phase)
                  bada_climb_alts = [alt_start]
                  current_alt = bada_climb_alts[0]
                  while current_alt < alt_cruise:
                        current_FL = current_alt / 100 * FT_TO_M
                        closest_FL = ac_data.index[np.abs(ac_data.index - current_FL).argmin()]
                        bada_vert_rate = ac_data.loc[closest_FL, "ROCDnom_cl"]
                        bada_climb_alts.append(current_alt)
                        current_alt += bada_timestep * bada_vert_rate * FT_TO_M / 60

                  # Go backwards from end altitude to cruise altitude (descent phase)
                  bada_decent_alts = [alt_stop]
                  current_alt = bada_decent_alts[0]
                  while current_alt < alt_cruise:
                        current_FL = current_alt / 100 * FT_TO_M
                        closest_FL = ac_data.index[np.abs(ac_data.index - current_FL).argmin()]
                        bada_vert_rate = ac_data.loc[closest_FL,"ROCDnom_des"]
                        bada_decent_alts.append(current_alt)
                        current_alt += bada_timestep * bada_vert_rate * FT_TO_M / 60

                  if len(bada_climb_alts) + len(bada_decent_alts) < len(bada_times):
                        break
                  else:
                        # If the climb and descent phases are too long, reduce the cruise altitude
                        alt_cruise -= DELTA_ALT_CRUISE
                        if alt_cruise < 0:
                              raise ValueError(f"Warning: Cruise altitude for {flight_path_name} is below 0. Skipping flight.")

            if alt_cruise != BADA_AC_CRUISE_LEVELS.loc[ac_type, "cr_fl"] * 100 * FT_TO_M:
                  print(f"Adjusted cruise altitude for {flight_path_name} from FL {BADA_AC_CRUISE_LEVELS.loc[ac_type, 'cr_fl']} to FL {int(alt_cruise / FT_TO_M / 100)} due to lack of time to reach cruise altitude.")

            num_points_cruise = len(bada_times) - (len(bada_climb_alts) + len(bada_decent_alts))
            bada_cruise_alts = np.linspace(alt_cruise, alt_cruise, num_points_cruise)
            bada_decent_alts = bada_decent_alts[::-1]
            bada_alts = np.concatenate([bada_climb_alts, bada_cruise_alts, bada_decent_alts])
            print(f"Length of BADA altitudes. Cruise: {len(bada_cruise_alts)}, Climb: {len(bada_climb_alts)}, Descent: {len(bada_decent_alts)}")

            time_series = (flight_path['time'] - flight_path["time"][0]) / 60**2
            plt.plot(bada_times, bada_alts, label=f"{ac_type} BADA {flight_path_name}", color=colors(i), linestyle='--')
            plt.scatter(time_series, flight_path['baroaltitude'], s=5, marker='x', color=colors(i), alpha=0.5)

      plt.legend()
      plt.xlabel("Time (hours)")
      plt.ylabel("Baro Altitude (m)")
      plt.show()


def vert_rate_vs_altitude(flight_paths):
      """
      Plot vertical rate (vertrate) vs barometric altitude for all flights and compare to BADA curves.
      """
      plt.figure(figsize=(10, 5))
      plotted_bada = set()
      all_baroaltitude = []
      all_vertrate = []
      for flight_path_name, flight_path in flight_paths.items():
            ac_type = flight_path_name.split("_")[2]
            ptf_filepath = f"BADA/{ac_type}__.ptf"
            ptf = read_ptf(ptf_filepath)
            ac_data = ptf['table']

            # Only keep one point every 60 seconds for plotting
            time_seconds = flight_path["time"] - flight_path["time"].iloc[0]
            mask = (time_seconds // 60).diff().fillna(1).astype(bool)
            sampled_baroaltitude = flight_path['baroaltitude'][mask]
            sampled_vertrate = flight_path["vertrate"][mask]
            all_baroaltitude.extend(sampled_baroaltitude)
            all_vertrate.extend(sampled_vertrate)

            # Only plot BADA curves once per aircraft type
            if ac_type not in plotted_bada:
                  plt.plot(ac_data.index * 100 * FT_TO_M, ac_data["ROCDhi_cl"] * FT_TO_M / 60, label="BADA High Load Climb Rate", c="red")
                  plt.plot(ac_data.index * 100 * FT_TO_M, ac_data["ROCDnom_cl"] * FT_TO_M / 60, label="BADA Nominal Load Climb Rate", c="limegreen")
                  plt.plot(ac_data.index * 100 * FT_TO_M, ac_data["ROCDlo_cl"] * FT_TO_M / 60, label="BADA Low Climb Load Rate", c="purple")
                  plt.plot(ac_data.index * 100 * FT_TO_M, ac_data["ROCDnom_des"] * FT_TO_M / -60, label="BADA Nominal Load Descent Rate", c="blue")
                  # Add idle descent rate curve (glide slope 17:1)
                  # Convert Vdes from knots to m/s before dividing by glide ratio
                  idle_descent_rate = (ac_data["Vdes"] * 0.514444) / 17  # Vdes in knots, convert to m/s, 17:1 glide slope
                  plt.plot(ac_data.index * 100 * FT_TO_M, -idle_descent_rate, label="17:1 Glide Slope", c="orange", linestyle="--")
                  plotted_bada.add(ac_type)


      plt.scatter(all_baroaltitude,all_vertrate,s=8,marker='.',label="Observed ROCD",alpha=0.05)

      # Add horizontal lines at 1 and -1 m/s for cruise cutoff
      plt.axhline(1, color='black', linestyle='--', linewidth=1.5, label='Cruise Cutoff')
      plt.axhline(-1, color='black', linestyle='--', linewidth=1.5)

      plt.xlabel("Barometric Altitude (m)")
      plt.ylabel("Vertical Rate (m/s)")
      # plt.title("Vertical Rate vs Barometric Altitude (All Flights)")
      plt.ylim(-30, 30)
      plt.xlim(0, 12500)
      plt.legend(fontsize=8)
      plt.show()




flight_paths = load_flight_paths(typecode_filter="A320", limit=10)
plot_overall(flight_paths, color_by="gs")
detail_plot(flight_paths)
baro_vs_geo_altitude_plot(flight_paths)
compare_with_bada(flight_paths)
vert_rate_vs_altitude(flight_paths)
