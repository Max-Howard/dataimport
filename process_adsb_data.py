import os
import pandas as pd
import numpy as np
import shutil
import xarray as xr
import  multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Non flight data
MET_DATA_DIR: str = "./MetData/"
MET_DATA = None
AIRPORT_DATA_PATH: str = "./airports.csv"
AIRPORT_DATA = None

# Flight data
BASE_DIR = "./FlightData/"
INPUT_DIR = os.path.join(BASE_DIR, "Raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "Processed")

# Tolerances to determine considered anomalous points
V_MAX = 1000         # Max velocity (m/s)
ROCD_MAX = 50        # Max rate of climb/descent (m/s)

# Tolerance for removing points
RDP_EPSILON = 10          # RDP tolerance in (m)

# Tolerance for dropping flights
MAX_TIME_GAP = 60           # Maximum time gap in seconds between points (s)
TOL_DIST_START_END = 10000  # Max distance flight can start/end from apt (m)
TOL_ALT_START_END = 1000    # Max height above airport at start and end of data (m)

MAX_WORKERS = multiprocessing.cpu_count()-1


def find_flight_files(directory: str):
    flight_files = []
    for file in os.listdir(directory):
        if file.endswith(".csv") or file.endswith(".pkl"):
            flight_files.append(file)
    return flight_files


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run basic cleaning steps on the DataFrame and sort by time.
    """
    if "lastposupdate" in df.columns:
        raise ValueError("DataFrame contains 'lastposupdate' column, this should be used as time.")
    df.dropna(inplace=True)
    df.drop_duplicates(subset=["time"], keep="first", inplace=True)
    df.sort_values(by="time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def remove_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where each of lat, lon, baroaltitude, geoaltitude
    don't cause velocity or rate of climb/descent to exceed the given tolerances.
    This should be run after the RDP simplification, as is itterative and therefore slow.
    NOTE this will break if the initial point is anomalous.
    """

    if "lastposupdate" in df.columns:
        raise ValueError("DataFrame contains 'lastposupdate' column, this should be used as time.")

    kept_idx = []
    last_vals = {}
    # before = len(df)
    for idx, row in df.iterrows():
        lat, lon = row["lat"], row["lon"]
        baroaltitude, geoaltitude = row["baroaltitude"], row["geoaltitude"]
        time = row["time"]

        if not last_vals: # Setup initial values
            kept_idx.append(idx)
            last_vals = dict(lat=lat, lon=lon, baroaltitude=baroaltitude, geoaltitude=geoaltitude, time=time)
            continue

        d_lat  = abs(lat  - last_vals["lat"])
        d_lon  = abs(lon  - last_vals["lon"])
        d_baroaltitude = abs(baroaltitude - last_vals["baroaltitude"])
        d_geoaltitude  = abs(geoaltitude  - last_vals["geoaltitude"])
        d_time = time - last_vals["time"]

        tol_lat = V_MAX * d_time / 111000
        tol_lon = V_MAX * d_time / (111000 * np.cos(np.radians(last_vals["lat"])))
        tol_alt = ROCD_MAX * d_time

        if (d_lat  <= tol_lat  and
            d_lon  <= tol_lon and
            d_baroaltitude <= tol_alt and
            d_geoaltitude  <= tol_alt):
            kept_idx.append(idx)
            last_vals = dict(lat=lat, lon=lon, baroaltitude=baroaltitude, geoaltitude=geoaltitude, time=time)
    # print(f"Removed {before - len(kept_idx)} points due to anomalies.")
    return df.loc[kept_idx].reset_index(drop=True)


def round_values(df: pd.DataFrame) -> pd.DataFrame:
    df['time'] = df['time'].round(2)
    df['lat'] = df['lat'].round(6)
    df['lon'] = df['lon'].round(6)
    df['geoaltitude'] = df['geoaltitude'].round(1)
    df['baroaltitude'] = df['baroaltitude'].round(1)
    df['gs'] = df['gs'].round(1)
    df['heading'] = df['heading'].round(1)
    df['vertrate'] = df['vertrate'].round(1)

    if 'tas' in df.columns:
        df['tas'] = df['tas'].round(1)
    if 'wind_speed' in df.columns:
        df['wind_speed'] = df['wind_speed'].round(1)
    if 'wind_dir' in df.columns:
        df['wind_dir'] = df['wind_dir'].round(1)

    # Rounding can re-introduce duplicates - drop them
    df.drop_duplicates(subset=["time"], keep="first", inplace=True)

    return df


def calc_dist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the cumulative distance along the trajectory using the Haversine formula and adds a 'dist' column.
    """
    lat = np.radians(df['lat'].to_numpy())
    lon = np.radians(df['lon'].to_numpy())

    delta_lat = lat[1:] - lat[:-1]
    delta_lon = lon[1:] - lon[:-1]

    a = (np.sin(delta_lat/2)**2 +
         np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(delta_lon/2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = np.concatenate(([0], 6371e3 * c))
    cumulative_dist = np.cumsum(dist)
    df['dist'] = np.round(cumulative_dist, 1)
    return df


def calc_tas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the True Airspeed (TAS) using the wind speed and direction from the MET data.
    Uses barometric altitude to find wind data, as Merra altitudes are calculated from pressure.
    Points are grouped by corresponding Met cell to reduce number of lookups.
    """

    if MET_DATA is None:
        raise ValueError("MET data has not been loaded.")

    # TODO create check to ensure MET_DATA is in range

    # Find the unique lat, lon, time, and baroaltitude combinations
    # TODO this is calculating based off of edges, not centers. Can be improved.
    df["lat_idx"] = np.digitize(df["lat"], MET_DATA["lat"].values)
    df["lon_idx"] = np.digitize(df["lon"], MET_DATA["lon"].values)
    df["time_idx"] = np.digitize(df["time"], MET_DATA["time"].values.astype(np.int64)/1e9) # TODO this may not be selecting correct time
    df["lev_idx"]  = np.digitize(df["baroaltitude"], MET_DATA["h_edge"].values) - 1 # TODO this is a temporary fix
    unique_groups = df[["lat_idx", "lon_idx", "time_idx", "lev_idx"]].drop_duplicates().reset_index(drop=True)

    # DataArrays are created for vectorized indexing
    da_lat = xr.DataArray(unique_groups["lat_idx"].values, dims="points")
    da_lon = xr.DataArray(unique_groups["lon_idx"].values, dims="points")
    da_time = xr.DataArray(unique_groups["time_idx"].values, dims="points")
    da_lev = xr.DataArray(unique_groups["lev_idx"].values, dims="points")

    # Wind speed and direction looked up and merged into original DataFrame and intermediate columns dropped
    unique_groups["wind_speed"] = MET_DATA["WS"].isel(lev=da_lev, lat=da_lat, lon=da_lon, time=da_time).values
    unique_groups["wind_dir"] = MET_DATA["WDIR"].isel(lev=da_lev, lat=da_lat, lon=da_lon, time=da_time).values
    df = df.merge(unique_groups, on=["lat_idx", "lon_idx", "time_idx", "lev_idx"], how="left")
    df.drop(columns=["lat_idx", "lon_idx", "time_idx", "lev_idx"], inplace=True)

    # TAS calculated from ground speed, wind speed, and wind direction
    df["tas"] = np.sqrt(df["gs"]**2 + df["wind_speed"]**2 -
        2 * df["gs"] * df["wind_speed"] * np.cos(np.radians(df["wind_dir"] - df["heading"])))

    return df


def create_save_dir():
    if os.path.exists(OUTPUT_DIR):
        remove_dir = input(f"The directory {OUTPUT_DIR} already exists. Do you want to overwrite it? (Y/n): ").strip().lower()
        if remove_dir == 'yes':
            shutil.rmtree(OUTPUT_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        with open(os.path.join(OUTPUT_DIR, ".gitignore"), "w") as gitignore:
            gitignore.write("*\n")


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = np.radians(lon2 - lon1)
    a = np.sin(d_phi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(d_lambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def rdp(points: np.ndarray, epsilon: float) -> list:
    """
    Ramer-Douglas-Peucker polyline simplification.
    points: np.ndarray of shape (n, m) where n is number of points and m
        is the number of dimensions (e.g., 3 for lat, lon, baroaltitude)
    epsilon: distance tolerance (same units as points)
    returns: list of indices to keep
    """
    # Base case
    if len(points) < 3:
        return [0, len(points) - 1]

    start, end = points[0], points[-1]
    line_vec = end - start
    line_len2 = np.dot(line_vec, line_vec)
    rel = points - start
    t = np.dot(rel, line_vec) / line_len2
    t = np.clip(t, 0.0, 1.0)
    proj = start + np.outer(t, line_vec)
    dists = np.linalg.norm(points - proj, axis=1)

    idx = np.argmax(dists)
    max_dist = dists[idx]

    if max_dist > epsilon:
        left = rdp(points[:idx+1], epsilon)
        right = rdp(points[idx:], epsilon)
        return left[:-1] + [i + idx for i in right]
    else:
        return [0, len(points) - 1]


def simplify_trajectory(df: pd.DataFrame, max_gap: float) -> pd.DataFrame:
    """
    Discared points that are within epsilon meters of the line segment between the first and last point.
    Ensures a point is kept a point every MAX_TIME_GAP seconds, as long as large gap does not already exist.
    """

    # Simple method for converting to meters ok here as distances are small and not used in final simulaton
    coords = np.vstack([
        df['lat'].to_numpy() * 111000,
        df['lon'].to_numpy() * 111000 * np.cos(np.radians(df['lat'].to_numpy())),
        df['baroaltitude'].to_numpy(),
        df['geoaltitude'].to_numpy()
    ]).T

    keep_idx = rdp(coords, RDP_EPSILON)
    keep_mask = np.zeros(len(df), dtype=bool)
    keep_mask[keep_idx] = True

    last_time = None
    for i, t in enumerate(df['time']):
        if last_time is None:
            last_time = t
            keep_mask[i] = True
        elif keep_mask[i]:
            last_time = t
        elif t - last_time >= max_gap:
            keep_mask[i-1] = True
            last_time = df['time'][i-1]

    df = df.iloc[keep_mask].reset_index(drop=True)
    return df


def process_file(flight_file_path: str):
    # print(f"Processing {flight_file_path}...")

    df = pd.read_csv(os.path.join(INPUT_DIR, flight_file_path))
    origin, destination, typecode, icao24, flight_number = flight_file_path.strip(".csv").split("_")

    df.rename(columns={"lastposupdate": "time", "velocity": "gs"}, inplace=True)
    df = clean_dataset(df)

    if len(df) < 1000:
        return {"file": flight_file_path, "status": "fail_insufficient_data", "len": len(df)}
    
    if df["time"].diff().max() > MAX_TIME_GAP:
        return {"file": flight_file_path, "status": "fail_patchy_data_pre_filter", "len": len(df)}

    # First simplification pass to allow for rapid removal of anomalous points is this is a slow process (iterative)
    # Max time gap is set to quarter the final value to ensure gaps are not too large after anomalies are removed
    pre_rdp_len = len(df)
    df = simplify_trajectory(df, MAX_TIME_GAP/4)
    df = remove_anomalies(df)
    # Second pass to remove excess points after removal of anomalies
    df = simplify_trajectory(df, MAX_TIME_GAP)

    if df["time"].diff().max() > MAX_TIME_GAP:
        return {"file": flight_file_path, "status": "fail_patchy_data_post_filter", "len": len(df)}

    if df["baroaltitude"].max() < 2000:
        return {"file": flight_file_path, "status": "fail_low_altitude", "len": len(df)}

    origin_airport_data = AIRPORT_DATA.loc[origin]
    destination_airport_data = AIRPORT_DATA.loc[destination]

    # Check if flight starts and ends near airport
    start_dist = haversine(df.iloc[0]['lat'], df.iloc[0]['lon'],origin_airport_data['lat'], origin_airport_data['lon'])
    end_dist = haversine(df.iloc[-1]['lat'], df.iloc[-1]['lon'],destination_airport_data['lat'], destination_airport_data['lon'])
    if start_dist > TOL_DIST_START_END or end_dist > TOL_DIST_START_END:
        return {"file": flight_file_path, "status": "fail_missing_start_end", "len": len(df)}
    start_height_above_apt = df["baroaltitude"].iloc[0] - origin_airport_data["alt"]
    end_height_above_apt = df["baroaltitude"].iloc[0] - destination_airport_data["alt"]
    if max(start_height_above_apt, end_height_above_apt) > TOL_ALT_START_END:
        return {"file": flight_file_path, "status": "fail_no_low_altitude_start_end", "len": len(df)}

    df = calc_dist(df)

    if df["dist"].iloc[-1] < 10000:
        return {"file": flight_file_path, "status": "fail_low_distance"}

    if CALC_TAS:
        df = calc_tas(df)
    df = round_values(df)

    df.to_csv(os.path.join(OUTPUT_DIR, flight_file_path), index=False)
    return {"file": flight_file_path, "status": "processed", "len": len(df), "rdp_dropped": pre_rdp_len - len(df)}


def init_worker(time_slice=None):
    """
    Worker initializer: load MET data (optionally with a time slice) into memory once per process.
    """
    global MET_DATA
    global AIRPORT_DATA
    AIRPORT_DATA = pd.read_csv("airports.csv").set_index("icao").copy()
    if CALC_TAS:
        ds = xr.open_dataset(os.path.join(MET_DATA_DIR, "wind_monthly_202411.nc4"))
        if time_slice is not None:
            ds = ds.isel(time=time_slice)
        MET_DATA = ds.load() # .load() significantly reduces io latency at the cost of ram usage


def print_results(results):
    if results:
        failed_patchy_pre = [r for r in results if r["status"] == "fail_patchy_data_pre_filtering"]
        failed_patchy_post = [r for r in results if r["status"] == "fail_patchy_data_post_filtering"]
        failed_length = [r for r in results if r["status"] == "fail_insufficient_data"]
        failed_low_altitude = [r for r in results if r["status"] == "fail_low_altitude"]
        failed_low_distance = [r for r in results if r["status"] == "fail_low_distance"]
        failed_no_low_altitude_start_end = [r for r in results if r["status"] == "fail_no_low_altitude_start_end"]
        failed_missing_start_end = [r for r in results if r["status"] == "fail_missing_start_end"]
        successful = [r for r in results if r["status"] == "processed"]
        num_points_successful = sum(r["len"] for r in successful)
        num_points_rdp_dropped = sum(r["rdp_dropped"] for r in successful)
        print(f"Number of files that failed due to patchy data pre filtering: {len(failed_patchy_pre)}")
        print(f"Number of files that failed due to patchy data post filtering: {len(failed_patchy_post)}")
        print(f"Number of files that failed due to insufficient data: {len(failed_length)}")
        print(f"Number of files that failed due to low max altitude: {len(failed_low_altitude)}")
        print(f"Number of files that failed due to low distance: {len(failed_low_distance)}")
        print(f"Number of files that failed due to no low altitude at start/end: {len(failed_no_low_altitude_start_end)}")
        print(f"Number of files that failed due to missing start/end: {len(failed_missing_start_end)}")
        print(f"Number of files processed successfully: {len(successful)}")
        print(f"Number of points processed successfully: {num_points_successful}")
        print(f"Number of points dropped due to RDP: {num_points_rdp_dropped}")

CALC_TAS = True
# Time slice for MET data to reduce memory usage, crucial when multiprocessing as each worker loads the data separately
# NOTE This should be set to a slice that covers the time period of the flights being processed
TIME_SLICE = slice(0, 16)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    create_save_dir()
    flight_file_paths = find_flight_files(INPUT_DIR)

    results = []
    print("Setting up multiprocessing, progress bar may hang for a moment...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker, initargs=(TIME_SLICE,)) as executor:
        for result in tqdm(executor.map(process_file, flight_file_paths), total=len(flight_file_paths), desc="Processing flights", unit="flight"):
            results.append(result)

    print_results(results)