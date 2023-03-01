# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 19:08:16 2023

@author: robin.marcille - mthiebau0107
"""

from sklearn.decomposition import PCA
import numpy as np
import xarray as xr
from numpy import ma
import netCDF4 as nc
import pandas as pd

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


def read_from_nc_files(years = ['2016', '2017', '2018']):
    """
    Extract MeteoNet data from nc files
    Mask continental points
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
    u = u*mask
    v = v*mask  

    return {'u': u, 'v': v}, lat, lon, mask


def compute_EOFs(u, v, lat, lon, mask, N_eofs):
    # reshape in 2D (time, space)
    Xu = np.reshape(u, (u.shape[0]*u.shape[1], len(lat)*len(lon)), order = 'F')
    Xv = np.reshape(v, (v.shape[0]*v.shape[1], len(lat)*len(lon)), order = 'F')

    # mask the land points
    Xu = ma.masked_array(Xu, np.isnan(Xu))
    Xv = ma.masked_array(Xv, np.isnan(Xv))

    land = Xu.sum(0).mask
    sea = ~land

    # Keep only sea grid-points
    Xu = Xu[:, sea]
    Xv = Xv[:, sea]

    # Get rid of remaining NaN points in v
    df = pd.DataFrame(Xv)
    df = df.fillna(value = None, method = 'backfill')
    df = df.fillna(value = None, method = "ffill")
    Xv = np.asarray(df)

    # Standardize variables using the fit and transform methods of sklearn.preprocessing.scaler.StandardScaler
    Xu_scaler = sc.fit_transform(Xu)
    Xv_scaler = sc.fit_transform(Xv)

    ## EOF decomposition
    ## instantiates the PCA object define the limit to compute the EOFs
    lim = 0.90

    pca_Xu = PCA(n_components = lim)
    pca_Xv = PCA(n_components = lim)

    # fit
    pca_Xu.fit(Xu_scaler)
    pca_Xv.fit(Xv_scaler)

    ## keep number of PC sufficient to explain lim = 99 % of the original variance
    ipc_Xu = np.where(pca_Xu.explained_variance_ratio_.cumsum() >= lim)[0][0]
    ipc_Xv = np.where(pca_Xv.explained_variance_ratio_.cumsum() >= lim)[0][0]

    pca_score_Xu = pca_Xu.explained_variance_ratio_
    pca_score_Xv = pca_Xv.explained_variance_ratio_

    ## The Principal Components (PCs) are obtained by using the transform method of the pca object (pca_Xi_scaler)
    PCs_Xu = pca_Xu.transform(Xu_scaler)
    Pcs_Xu = PCs_Xu[:,:ipc_Xu]

    PCs_Xv = pca_Xv.transform(Xv_scaler)
    Pcs_Xv = PCs_Xv[:,:ipc_Xv]

    # The Empirical Orthogonal Functions (EOFs) are contained in the components_ attribute of the pca object (pca_Xi_scaler)
    EOFs_Xu = pca_Xu.components_
    EOFs_Xv = pca_Xv.components_

    EOFs_Xu = EOFs_Xu[:ipc_Xu,:]
    EOFs_Xv = EOFs_Xv[:ipc_Xv,:]

    ### Recontruction of the 2D fields
    EOF_recons_Xu = np.ones((ipc_Xu, len(lat) * len(lon))) * -999.
    EOF_recons_Xv = np.ones((ipc_Xv, len(lat) * len(lon))) * -999.

    for i in range(ipc_Xu): 
        EOF_recons_Xu[i,sea] = EOFs_Xu[i,:]
        
    for i in range(ipc_Xv): 
        EOF_recons_Xv[i,sea] = EOFs_Xv[i,:]
        
    EOF_recons_Xu = ma.masked_values(np.reshape(EOF_recons_Xu, (ipc_Xu, len(lat), len(lon)), order='F'), -999.)
    EOF_recons_Xv = ma.masked_values(np.reshape(EOF_recons_Xv, (ipc_Xv, len(lat), len(lon)), order='F'), -999.)
    
    return {'u' : EOF_recons_Xu, 'v' : EOF_recons_Xv}

def read_eofs(EOFs, mask, N_eofs, lat_flat):
    EOFs_u = EOFs['u'][:N_eofs, :, :] * mask
    EOFs_v = EOFs['v'][:N_eofs, :, :] * mask
    EOFs_u = np.ma.masked_array(EOFs_u, np.isnan(EOFs_u))
    EOFs_v = np.ma.masked_array(EOFs_v, np.isnan(EOFs_v))
    sea2 = ~EOFs_u.mask
    EOFs_v = np.reshape(EOFs_v[sea2], (N_eofs, len(lat_flat)))
    EOFs_u = np.reshape(EOFs_u[sea2], (N_eofs, len(lat_flat)))
    V_svd = pd.concat([pd.DataFrame(EOFs_u.T), pd.DataFrame(EOFs_v.T)], axis = 1)

    EOFs = np.vstack((EOFs_u, EOFs_v))

    return EOFs_u, EOFs_v, EOFs, V_svd