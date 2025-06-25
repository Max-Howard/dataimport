"""
Script to prepare wind data for openAVEM from MERRA-2 data
"""

import os
import xarray as xr
import numpy as np
import glob
from datetime import datetime, timezone

BASEDIR = "WindData/"

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
    H_EDGES = np.array(
        [-6, 123, 254, 387, 521, 657, 795, 934, 1075, 1218, 1363, 1510, 1659,
         1860, 2118, 2382, 2654, 2932, 3219, 3665, 4132, 4623, 5142, 5692,
         6277, 6905, 7582, 8320, 9409, 10504, 11578, 12633]
    )
    dslist = []
    for fpath in files_in:
        print(fpath)
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
        ds = ds.mean('time')
        ds = ds.assign(WS=np.sqrt(ds.U**2 + ds.V**2))
        ds.WS.attrs = {'long_name': 'wind_speed',
                       'standard_name': 'wind_speed',
                       'units': 'm s-1',}
        ds = ds.assign(WDIR=np.arctan2(ds.U, ds.V)*180/np.pi)
        ds.WDIR.attrs = {'long_name': 'wind_direction',
                         'standard_name': 'wind_direction',
                         'units': 'degrees clockwise from north',}
        ds = ds.drop(['U', 'V'])
    
    # Add altitude coordinate
    ds.coords['h_edge'] = ('lev', H_EDGES,
                           {'long_name': ('altitude over sea level'
                                          + ' at lower edge of grid box'),
                            'units': 'm'})
        
    now = datetime.now(timezone.utc).strftime('%Y/%m/%d %H:%M:%S %Z')
    ds.attrs = {
        'Title': 'MERRA2 time-averaged wind, processed from GEOS-Chem input'
                 + ' files for use in openAVEM',
         'Contact': 'Flávio Quadros (f.quadros@tudelft.nl)',
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
                    chunksizes=[1, ds.lat.size, ds.lon.size])
    enc = {v:encoding for v in list(ds.data_vars.keys())}
    ds.to_netcdf(file_out, format='NETCDF4', engine='netcdf4',
                 encoding=enc)
    ds.close()
    print(' done')
    print(ds)
    
    return ds


def merge_months(year='2019', file_out='./met/wind_yearly_2019.nc4'):
    """
    Calculate yearly wind average from monthly averages

    Parameters
    ----------
    year : str, optional
        Year to average. The default is '2019'.
    file_out : str, optional
        Path to output file. The default is './met/wind_yearly_2019.nc4'.

    Returns
    -------
    ds : xr.Dataset
        Dataset containing wind speed [m/s] and wind direction [degrees
        clockwise from north].

    """
    search = f'./met/wind_monthly_{year}??.nc4'
    fpaths = glob.glob(search)
    totaldays = 0
    dslist = []
    for fpath in fpaths:
        ds = xr.open_dataset(fpath)
        ndays = int(ds.attrs['End_Date']) - int(ds.attrs['Start_Date']) + 1
        totaldays += ndays
        
        with xr.set_options(keep_attrs=True):
            ds = ds.assign(U = ds['WS'] * np.sin(ds['WDIR'] * np.pi / 180) * ndays)
            ds = ds.assign(V = ds['WS'] * np.cos(ds['WDIR'] * np.pi / 180) * ndays)
            ds.drop(['WS', 'WDIR'])
        
        dslist.append(ds)
    
    ds = dslist[0].copy()
    print(0)
    for i in range(1, len(dslist)):
        ds += dslist[i]
        print(i)
    ds /= totaldays
    print('Processing...', end='')
    old_attrs = dslist[0].attrs
    print(dslist[0].attrs)
    start_date = min([d.attrs['Start_Date'] for d in dslist])
    end_date = max([d.attrs['End_Date'] for d in dslist])
    with xr.set_options(keep_attrs=True):
        ds = ds.assign(WS=np.sqrt(ds.U**2 + ds.V**2))
        ds.WS.attrs = {'long_name': 'wind_speed',
                       'standard_name': 'wind_speed',
                       'units': 'm s-1',}
        ds = ds.assign(WDIR=np.arctan2(ds.U, ds.V)*180/np.pi)
        ds.WDIR.attrs = {'long_name': 'wind_direction',
                         'standard_name': 'wind_direction',
                         'units': 'degrees clockwise from north',}
        ds = ds.drop(['U', 'V'])
    
    now = datetime.now(timezone.utc).strftime('%Y/%m/%d %H:%M:%S %Z')
    ds.attrs = {
        'Title': 'MERRA2 time-averaged wind, processed from GEOS-Chem input'
                 + ' files for use in openAVEM',
         'Contact': 'Flávio Quadros (f.quadros@tudelft.nl)',
         'References': '',
         'Filename': os.path.basename(file_out),
         'History': 'File generated on: ' + now,
         'ProductionDateTime': 'File generated on: ' + now,
         'ModificationDateTime': 'File generated on: ' + now,
         'Format': 'NetCDF-4',
         'SpatialCoverage': 'global',
         'Version': 'MERRA2',
         'VersionID': old_attrs['VersionID'],
         'Nlayers': len(ds.lev),
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
                    chunksizes=[1, ds.lat.size, ds.lon.size])
    enc = {v:encoding for v in list(ds.data_vars.keys())}
    ds.to_netcdf(file_out, format='NETCDF4', engine='netcdf4',
                 encoding=enc)
    ds.close()
    print(' done')
    print(ds)
    
    return ds


if __name__ == '__main__':

    year:str = '2024'
    month_start:int = 6
    month_end:int = 12

    for month in [f'{m:02}' for m in range(month_start, month_end + 1)]:
        output_fpath = BASEDIR + r'./ForOpenAVEM/wind_monthly_' + year + month + '.nc4'
        search = BASEDIR + f'Raw/{year}/{month}/' + r'*.A3dyn.*.nc4'
        fpaths = glob.glob(search)
        if len(fpaths) > 0:
            ds = process_files(fpaths, output_fpath)
        else:
            print(f'Warning: no files found for "{search}"')
