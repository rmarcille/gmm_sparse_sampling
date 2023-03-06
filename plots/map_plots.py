# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 19:08:16 2023

@author: robin.marcille
"""

import cartopy.crs as ccrs
import cartopy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_EOF(EOF, lat, lon, n_eof, param):
    """plot EOF example

    Args:
        EOF (numpy array): Empirical Orthogonal functions (Concatenated u and v) (n EOFs, grid points)
        lat (numpy array): latitudes of grid points (1D)
        lon (numpy array): longitudes of grid points (1D)
        n_eof (_type_): number of EOF to plot
        param (_type_): parameter to plot (u / v)
    """

    df_eof = pd.DataFrame({'param' : EOF[n_eof, :], 'lat' : lat, 'lon' : lon})
    eof_map = pd.pivot_table(df_eof, values = 'param', index = 'lat', columns = 'lon')
    lat = pd.DataFrame(lat).round(3)
    lon = pd.DataFrame(lon).round(3)
    minlon = lon.min()[0]
    minlat = lat.min()[0]
    maxlon = lon.max()[0]
    maxlat = lat.max()[0]
    ax0 = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())

    plt.xlim((minlon,maxlon+0.01))
    plt.ylim((minlat-0.01, maxlat))

    ax0.coastlines(zorder = 3)
    ax0.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black')
    XLON, YLAT = np.meshgrid(eof_map.columns, eof_map.index)
    vmin = np.min(EOF[n_eof,:])
    vmax = np.max(EOF[n_eof,:])
    cs = plt.contourf(XLON, YLAT, eof_map,
                        levels = np.linspace(vmin, vmax, 20), 
                        vmin = vmin, vmax = vmax, cmap = 'bone')
    cbar = plt.colorbar(cs, ticks = np.round(np.linspace(vmin+0.005,vmax-0.005,5), 3), location = 'top')
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('EOF n°' + str(n_eof) + ' - ' + param, fontsize = 13)
    ax0.set_ylabel('Latitude [°N]', fontsize = 16)
    ax0.set_xlabel('Longitude [°E]', fontsize = 16)
    plt.xticks([3, 4, 5, 6])
    plt.yticks([43, 43.5]) 
    ax0.tick_params(axis='y', labelsize=16, direction = 'out', length = 3, color = 'k', width = 1)


def plot_sensors_locations(lon_flat, lat_flat, idx_sensors, EOF, n_eof = 0):
    """Show sensors locations for different methods, with EOF of zonal or meridional wind speed as background

    Args:
        lon_flat (np array): longitudes of grid points (1D)
        lat_flat (np array): latitudes of grid points (1D)
        idx_sensors (dict): dict of lists of sensors locations
        EOF (numpy array): Empirical Orthogonal functions (Concatenated u and v) (n EOFs, grid points)
        n_eof (int, optional): number of EOF to plot. If n_eof > EOF.shape[0]/2, parameter v is considered. Defaults to 0.
    """
    if n_eof < 10:
        param = 'u'
    df_eof = pd.DataFrame({'param' : EOF[n_eof, :], 'lat' : lat_flat, 'lon' : lon_flat})
    eof_map = pd.pivot_table(df_eof, values = 'param', index = 'lat', columns = 'lon')

    xticks = [3, 4, 5, 6]
    yticks = [43, 43.5]
    n_methods = 4
    fig, ax0 = plt.subplots(3, 1, figsize = (15,10))
    name_area = 'Mediterranean Sea'
    methods = ['GMM', 'QR', 'EOF_extrema']
    for j, method in enumerate(methods): 
        ax0 = plt.subplot(3, 1, j+1, projection=ccrs.PlateCarree())
        ax0.coastlines()
        ax0.add_feature(cartopy.feature.OCEAN, zorder=0)
        ax0.add_feature(cartopy.feature.LAND, zorder=2, edgecolor='black')
        ax0.set_ylabel('Latitude [°N]', fontsize = 16)
        plt.xticks(xticks)
        ax0.tick_params(axis='x', labelsize=16, direction = 'out', length = 3, color = 'k', width = 1)
        if j == n_methods-1:
            ax0.set_xlabel('Longitude [°E]', fontsize = 16)
        plt.yticks(yticks) 
        ax0.tick_params(axis='y', labelsize=16, direction = 'out', length = 3, color = 'k', width = 1)
        if j == 0 : 
            ax0.set_title(name_area, fontsize = 16)

        ax0.scatter(lon_flat[idx_sensors[method]], lat_flat[idx_sensors[method]], c = 'r', s = 100,  marker = 'o', zorder = 5)

        XLON, YLAT = np.meshgrid(eof_map.columns, eof_map.index)
        vmin = np.min(EOF[n_eof,:])
        vmax = np.max(EOF[n_eof,:])
        cs = plt.contourf(XLON, YLAT, eof_map,
                            levels = np.linspace(vmin, vmax, 20), 
                            vmin = vmin, vmax = vmax, cmap = 'bone')
        cbar = plt.colorbar(cs, ticks = np.round(np.linspace(vmin+0.005,vmax-0.005,5), 3), location = 'right')
        cbar.ax.tick_params(labelsize=13)
        cbar.set_label('EOF n°' + str(n_eof) + ' - ' + param, fontsize = 13)
        ax0.text(6, 43.5, method,
                 horizontalalignment='right',
                 verticalalignment='center', fontsize = 16)
