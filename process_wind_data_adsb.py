"""
Script to prepare wind data for openAVEM from MERRA-2 data
"""

import os
import xarray as xr
import numpy as np
import glob
from datetime import datetime, timezone
import calendar


BASEDIR = 'WindData/'


def process_files(files_in, file_out, nlayers=32):
    """
    Read files_in, concatenate, and save file_out with wind speed and direction

    Parameters
    ----------
    files_in : list of str
        Paths to the netCDF files containing MERRA-2 wind data.
    file_out : str
        Path to output file.
    nlayers : int or None, optional
        Number of vertical layers to keep. If None, all layers are kept.
        The default is 32.

    Returns
    -------
    ds : xr.Dataset
        Dataset containing wind speed [m/s] and wind direction [degrees
        clockwise from north].

    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(file_out)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Create a .gitignore file if it doesn't exist
    gitignore_path = os.path.join(output_dir, '.gitignore')
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, 'w') as gitignore_file:
            gitignore_file.write('*\n')
    H_EDGES = np.array(
        [-6, 123, 254, 387, 521, 657, 795, 934, 1075, 1218, 1363, 1510, 1659,
         1860, 2118, 2382, 2654, 2932, 3219, 3665, 4132, 4623, 5142, 5692,
         6277, 6905, 7582, 8320, 9409, 10504, 11578, 12633]
    )
    dslist = []
    for fpath in files_in:
        # print(fpath)
        ds = xr.open_dataset(fpath, drop_variables=['DTRAIN', 'OMEGA', 'RH'])
        with xr.set_options(keep_attrs=True):
            ds = ds.sel(lev=slice(None, nlayers))
        dslist.append(ds)
        ds.close()

    print('Concatenating...', end='')
    ds = xr.concat(dslist, 'time')
    print(' done')
    print('Processing...', end='')
    old_attrs = ds.attrs
    start_date = min([d.attrs['Start_Date'] for d in dslist])
    end_date = max([d.attrs['End_Date'] for d in dslist])
    with xr.set_options(keep_attrs=True):
        ds = ds.sel(lev=slice(None, nlayers))
        ds = ds.assign(WS=np.sqrt(ds.U**2 + ds.V**2))
        ds.WS.attrs = {'long_name': 'wind_speed',
                       'standard_name': 'wind_speed',
                       'units': 'm s-1',}
        ds = ds.assign(WDIR=np.arctan2(ds.U, ds.V)*180/np.pi)
        ds.WDIR.attrs = {'long_name': 'wind_direction',
                         'standard_name': 'wind_direction',
                         'units': 'degrees clockwise from north',}
        ds = ds.drop_vars(['U', 'V'])

    # Add altitude coordinate
    ds.coords['h_edge'] = ('lev', H_EDGES,
                           {'long_name': ('altitude over sea level'
                                          + ' at lower edge of grid box'),
                            'units': 'm'})

    now = datetime.now(timezone.utc).strftime('%Y/%m/%d %H:%M:%S %Z')
    ds.attrs = {
        'Title': 'MERRA2 concatenated wind data, processed from GEOS-Chem input'
                 + ' files for use in openAVEM',
         'Contact': 'Maximilian Howard (mdeh2@cam.ac.uk)',
         'References': '',
         'Filename': os.path.basename(file_out),
         'History': 'File generated on: ' + now,
         'ProductionDateTime': 'File generated on: ' + now,
         'ModificationDateTime': 'File generated on: ' + now,
         'Format': 'NetCDF-4',
         'SpatialCoverage': 'global',
         'Version': 'MERRA2',
         'VersionID': old_attrs['VersionID'],
         'Nlayers': nlayers,
         'Start_Date': start_date,
         'Start_Time': '00:00:00.0',
         'End_Date': end_date,
         'End_Time': '23:59:59.99999',
         'Delta_Lon': old_attrs['Delta_Lon'],
         'Delta_Lat': old_attrs['Delta_Lat']
    }
    print(' done')

    print(f'Saving "{file_out}"...', end='')
    encoding = dict(zlib=True, shuffle=True, complevel=1,
                    chunksizes=[ds.time.size, 1, ds.lat.size, ds.lon.size])
    enc = {v:encoding for v in list(ds.data_vars.keys())}
    ds.to_netcdf(file_out, format='NETCDF4', engine='netcdf4',
                 encoding=enc)
    ds.close()
    print(' done')
    return ds

def check_files(files_in) -> bool:
    """
    Check files_in for missing attributes and variables

    Parameters
    ----------
    files_in : list of str
        Paths to the netCDF files containing MERRA-2 wind data.

    Returns
    -------
    bool
        True if any issues are found, False otherwise.

    """
    for fpath in files_in:
        ds = xr.open_dataset(fpath, decode_times=False)
        if np.max(ds["U"].isel(time=slice(0, 2), lev=slice(0, 2), lat=slice(0, 2), lon=slice(0, 2)).values) == 0:
            return True
        for var in ['U', 'V']:
            if var not in ds.data_vars:
                print(f'\nWARNING: variable "{var}" not found in "{fpath}\n"')
                return True
        for time in [90, 270, 450, 630, 810, 990, 1170, 1350]:
            if time not in ds['time']:
                print(f'\nWARNING: Incorrect time axis in "{fpath}"\n')
                return True
        for attr in ['Start_Date', 'End_Date', 'VersionID', 'Delta_Lon', 'Delta_Lat']:
            if attr not in ds.attrs:
                print(f'\nWARNING: attribute "{attr}" not found in "{fpath}"\n')
                return True
        for dim, length in {'time': 8, 'lev': 72, 'lat': 361, 'lon': 576}.items():
            if dim not in ds.dims:
                print(f'\nWARNING: dimension "{dim}" not found in "{fpath}"\n')
                return True
            elif ds.sizes[dim] != length:
                print(f'\nWARNING: dimension "{dim}" has wrong size in "{fpath}"\n')
                return True
    return False



if __name__ == '__main__':
    year:str = '2024'
    month_start:int = 6
    month_end:int = 12

    # Process files
    for month in [f'{m:02}' for m in range(month_start, month_end + 1)]:
        output_fpath = BASEDIR + r'/ForADSB/wind_monthly_' + year + month + '.nc4'
        search = f'{BASEDIR}/Raw/{year}/{month}/' + r'*.A3dyn.*.nc4'
        fpaths = glob.glob(search)
        days_in_month = calendar.monthrange(int(year), int(month))[1]
        if len(fpaths) == 0:
            print(f'Warning: no files found for "{year}/{month}", skipping.')
        elif len(fpaths) != days_in_month:
            print(f'Warning: Expected {days_in_month} files, but found {len(fpaths)} for "{year}/{month}", skipping.')
        elif check_files(fpaths):
            print(f'Warning: Files for "{year}/{month}" have issues, skipping.')
        else:
            print(f'Processing {len(fpaths)} files for "{year}/{month}".')
            ds = process_files(fpaths, output_fpath)