# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 19:08:16 2023

@author: robin.marcille
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
import os
import datetime
import xarray as xr



def read_from_nc_files(years = ['2016', '2017', '2018']):
    """
    Extract MeteoNet data from nc files
    Mask continental points

    Args:
        years (list): list of years to read from

    Returns:
        (dict): Dictionnay containing the zonal and meridional wind speed data in 3D (time, lon, lat)
        df_med (pandas DataFrame) dataframe of zonal and meridional wind speed (time, grid points) concatenated
        lat (numpy array) list of latitudes
        lon (numpy array) list of longitudes
        mask (numpy masked array) land/sea mask
    """
    
    for i, year in enumerate(years):
        #Read the nc files of year using xarray
        ds = xr.open_mfdataset(f'./data/_Med/{year}/nc_files' + r'\*.nc', concat_dim='t', combine='nested')
        u = np.asarray(np.squeeze(ds.variables['u'][:]))    # u10m: eastward wind at 10m (zonal wind)
        v = np.asarray(np.squeeze(ds.variables['v'][:]))    # v10m: northward wind at 10m (meridional wind)

        #Initialize output
        if i == 0:
            u_out = u
            v_out = v

        #Concatenate in time
        else: 
            u_out = np.concatenate((u_out, u), axis = 0)
            v_out = np.concatenate((v_out, v), axis = 0)
        
    #Extract the land/sea mask
    dM = nc.Dataset('.\data\_Med\mask.nc')
    mask = dM['mask'][:]
    lat = dM['lat'][:]
    lon = dM['lon'][:]

    mask[mask==1] = float('nan')
    mask[mask==0] = 1
    
    #Mask land points
    u = u_out*mask
    v = v_out*mask  

    #Remove last time step (24 hours)
    u = u[:, :-1, :, :]
    v = v[:, :-1, :, :]
    u = np.ma.masked_array(u, np.isnan(u))
    v = np.ma.masked_array(v, np.isnan(v))

    #Update mask and reshape
    sea = ~u.mask
    N_sea = sea[0, 0, :, :].sum()
    u1d = np.reshape(u[sea], (u.shape[0]*u.shape[1], N_sea))
    v1d = np.reshape(v[sea], (v.shape[0]*v.shape[1], N_sea))
    u1d = pd.DataFrame(u1d)
    v1d = pd.DataFrame(v1d)
    df_med = pd.concat([u1d, v1d], axis = 1)
    n_input_points = u1d.shape[1]
    cols = ['u' + str(i) for i in range(n_input_points)] + ['v' + str(i) for i in range(n_input_points)]
    df_med.columns = cols

    return {'u': u, 'v': v}, df_med, lat, lon, mask

def process_time_index(years):
    """Read datetime index from nc files

    Args:
        years (list): list of years to read from

    Returns:
        time_index (pandas DatetimeIndex): pandas index containing input datetimes
    """

    time_index = pd.DatetimeIndex([])
    for i, year in enumerate(years):
        for j,filename in enumerate(os.listdir(f'./data/_Med/{year}/nc_files/')):
            month = int(filename[5:7])
            day = int(filename[8:10])
            hours = [i for i in range(24)]
            dt_index = pd.DatetimeIndex([datetime.datetime(int(year), month, day, hour) for hour in hours])
            time_index = pd.Index.union(time_index, dt_index)
    return time_index

def process_lat_lon(lat, lon, mask):
    """Process latitude and longitude list to output datafram format of sea points and flattened arrays

    Args:
        lat (np masked array): list of latitude on the grid
        lon (np masked array): list of longitude on the grid
        mask (np masked array): land/sea mask

    Returns:
        lat_flat (np masked array): flattened and masked array of latitude
        lon_flat (np masked array): flattened and masked array of longitude
        mask_flat (np masked array): flattened mask
        df_latlon (pandas DataFrame): Latitude and longitude dataframe
    """
    #Create a meshgrid of latitude and longitude
    XLON, YLAT = np.meshgrid(lon, lat)

    #Mask land points
    XLON = XLON*mask
    YLAT = YLAT*mask

    #Flatten
    lon_flat = np.reshape(XLON, (len(lat)*len(lon)))
    lat_flat = np.reshape(YLAT, (len(lat)*len(lon)))
    mask_flat = np.reshape(mask, (len(lat)*len(lon)))

    #Mask land points
    lon_flat = np.ma.masked_array(lon_flat, np.isnan(lon_flat))
    lat_flat = np.ma.masked_array(lat_flat, np.isnan(lat_flat))
    land = lon_flat.mask
    sea = ~land
    lon_flat = lon_flat[sea].data
    lat_flat = lat_flat[sea].data

    #Create dataframe
    df_latlon = pd.DataFrame({'lat' : lat_flat, 'lon' : lon_flat})
    return lat_flat, lon_flat, mask_flat, df_latlon
