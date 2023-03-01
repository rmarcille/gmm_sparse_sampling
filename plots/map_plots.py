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
    ax0.tick_params(axis='y', labelsize=16, direction = 'out', length = 3, color = 'k', width = 1)
