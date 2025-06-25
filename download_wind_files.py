import os
import glob
import requests
import urllib.parse
import xarray as xr
import numpy as np
from bs4 import BeautifulSoup
from bs4 import Tag
from datetime import datetime

# Base directory for storing downloaded files
SAVEDIR = os.path.join(os.path.dirname(__file__), 'WindData', "Raw")

def download_files(year, month) -> str:
    """
    Download MERRA-2 wind data files for a given year and month.
    The files are saved in a structured directory by year and month in SAVEDIR.
    Parameters
    ----------
    year : str
        The year for which to download the data (e.g., '2024').
    month : str
        The month for which to download the data (e.g., '01' for January).
    Returns
    -------
    save_dir : str
        The directory where the downloaded files are saved.
    """

    base_url = f"http://geoschemdata.wustl.edu/ExtData/GEOS_0.5x0.625/MERRA2/{year}/{month}/"
    save_dir = os.path.join(SAVEDIR, year, month)

    # Check download directory before starting
    existing_files = []
    if os.path.exists(save_dir):
        part_files = glob.glob(os.path.join(save_dir, '*.part'))
        for part_file in part_files:
            os.remove(part_file)
            print(f"Removed incomplete file: {part_file}")
        existing_files = [os.path.basename(f) for f in glob.glob(os.path.join(save_dir, '*.nc4'))]
        if existing_files:
            print(f"Found {len(existing_files)} existing files for {year}/{month}, these will not be redownloaded.")
    else:
        os.makedirs(save_dir)

    # Create a .gitignore file if it doesn't exist
    gitignore_path = os.path.join(SAVEDIR, '.gitignore')
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, 'w') as gitignore_file:
            gitignore_file.write('*\n')

    # Send a GET request to fetch the directory listing
    response = requests.get(base_url)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve the webpage. Status code: {response.status_code}")

    # Find the links to the files
    soup = BeautifulSoup(response.text, 'html.parser')
    all_links = soup.find_all('a')
    file_names = []
    for link in all_links:
        if isinstance(link, Tag):
            href = link.get('href')
            file_name: str = str(href) if href is not None else ""
            if file_name and "A3dyn" in file_name and file_name.endswith(".nc4"):
                if file_name in existing_files:
                    print(f"Skipping {file_name} as it already exists.")
                else:
                    file_names.append(file_name)

    print(f"{datetime.now().strftime('%H:%M:%S')}: Downloading {len(file_names)} MERRA-2 wind data files for {year}/{month}")

    # Download the files
    for idx, file_name in enumerate(file_names):
        file_response = requests.get(url=urllib.parse.urljoin(base_url, file_name), stream=True)
        file_num_string: str = f"{idx+1} of {len(file_names)}"
        if file_response.status_code == 200:
            start_time = datetime.now()
            file_path = os.path.join(save_dir, file_name)
            temp_file_path = file_path + ".part"
            total_size = int(file_response.headers.get('content-length', 0))
            block_size = 8192
            wrote = 0
            with open(temp_file_path, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=block_size):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
                        wrote += len(chunk)
                        done = int(50 * wrote / total_size)
                        print(f"\r{start_time.strftime('%H:%M:%S')}: Downloading file {file_num_string}."
                              + f" File size: {total_size / (1024 * 1024):.2f} MB [{'=' * done}{' ' * (50-done)}] {wrote / total_size:.2%}",
                              end='', flush=True)
            os.rename(temp_file_path, file_path)
            seconds_taken = int(round((datetime.now()-start_time).total_seconds(), 0))
            print("\r\033[K", end='', flush=True)
            print(f"File {file_num_string}. Downloaded {total_size / (1024 * 1024):.2f} MB in {seconds_taken}s. Saved to: {file_path}")
        else:
            raise Exception(f"Failed to download {file_name}. Status code: {file_response.status_code}")
    return save_dir

def check_files(files_in):
    """
    Check files_in for missing attributes and variables, as some of the files have been found to have issues.

    Parameters
    ----------
    files_in : list of str
        Paths to the netCDF files containing MERRA-2 wind data.

    Returns
    -------
    None.

    """
    for fpath in files_in:
        print(f'Checking "{fpath}"...')
        ds = xr.open_dataset(fpath, decode_times=False)
        if np.max(ds["U"].isel(time=slice(0, 2), lev=slice(0, 2), lat=slice(0, 2), lon=slice(0, 2)).values) == 0:
            print(f'\nWARNING: "{fpath}" contains only zeros in the checked slice\n')
            input("Press Enter to continue...")
        for var in ['U', 'V']:
            if var not in ds.data_vars:
                print(f'\nWARNING: variable "{var}" not found in "{fpath}\n"')
                input("Press Enter to continue...")
                break
        for time in [90, 270, 450, 630, 810, 990, 1170, 1350]:
            if time not in ds['time']:
                print(f'\nWARNING: Incorrect time axis in "{fpath}"\n')
                input("Press Enter to continue...")
                break
        for attr in ['Start_Date', 'End_Date', 'VersionID', 'Delta_Lon', 'Delta_Lat']:
            if attr not in ds.attrs:
                print(f'\nWARNING: attribute "{attr}" not found in "{fpath}"\n')
                input("Press Enter to continue...")
                break
        for dim, length in {'time': 8, 'lev': 72, 'lat': 361, 'lon': 576}.items():
            if dim not in ds.dims:
                print(f'\nWARNING: dimension "{dim}" not found in "{fpath}"\n')
                input("Press Enter to continue...")
                break
            elif ds.sizes[dim] != length:
                print(f'\nWARNING: dimension "{dim}" has wrong size in "{fpath}"\n')
                input("Press Enter to continue...")
                break

if __name__ == "__main__":
    # Example usage
    year = '2024'
    month = '01'
    save_dir = download_files(year, month)
    
    # Check the downloaded files
    files_in = glob.glob(os.path.join(save_dir, '*.nc4'))
    check_files(files_in)
    
    print(f"All files in {save_dir} have been checked.")