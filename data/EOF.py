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

def compute_EOFs(u, v, lat, lon, mask, idx_train, idx_test, N_eofs):
    """Compute EOFs of u and v fields

    Args:
        u (np masked array): masked array of zonal velocity
        v (np masked array): masked array of meridional velocity
        lat (numpy array): list of grid latitudes
        lon (numpy array): list of grid longitudes
        mask (np masked array): land/sea mask
        N_eofs (float): Number of EOFs to compute. If an int > 1, computes up to N_eofs for u and v. 
                        if N_eofs is a float < 1, computes EOFs so that the variance explained is == N_eofs

    Returns:
        dict: dictionnary containing the EOFs of u and v, and their 2D spatial reconstructions with masks
    """

    # reshape in 2D (time, space)
    Xu = np.reshape(u, (u.shape[0]*u.shape[1], len(lat)*len(lon)))
    Xv = np.reshape(v, (v.shape[0]*v.shape[1], len(lat)*len(lon)))

    #Keep training index only 
    Xu = Xu[idx_train, :]
    Xv = Xv[idx_train, :]
    
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
    pca_Xu = PCA(n_components = N_eofs)
    pca_Xv = PCA(n_components = N_eofs)

    # fit
    pca_Xu.fit(Xu_scaler)
    pca_Xv.fit(Xv_scaler)

    ## The Principal Components (PCs) are obtained by using the transform method of the pca object (pca_Xu_scaler)
    PCs_Xu = pca_Xu.transform(Xu_scaler)
    PCs_Xv = pca_Xv.transform(Xv_scaler)

    # The Empirical Orthogonal Functions (EOFs) are contained in the components_ attribute of the pca object (pca_Xu_scaler)
    EOFs_Xu = pca_Xu.components_
    EOFs_Xv = pca_Xv.components_

    if N_eofs < 1:
    ## N_eofs is the limit of variance explained to compute the EOFs
    ## keep number of PC sufficient to explain lim = 99 % of the original variance - Un-comment to run
        ipc_Xu = np.where(pca_Xu.explained_variance_ratio_.cumsum() >= N_eofs)[0][0]
        ipc_Xv = np.where(pca_Xv.explained_variance_ratio_.cumsum() >= N_eofs)[0][0]

        pca_score_Xu = pca_Xu.explained_variance_ratio_
        pca_score_Xv = pca_Xv.explained_variance_ratio_

        Pcs_Xu = PCs_Xu[:, :ipc_Xu]
        Pcs_Xv = PCs_Xv[:, :ipc_Xv]

        EOFs_Xu = EOFs_Xu[:ipc_Xu, :]
        EOFs_Xv = EOFs_Xv[:ipc_Xv, :]

    else:
        ipc_Xu = N_eofs
        ipc_Xv = N_eofs

    ### Recontruction of the 2D fields
    EOF_recons_Xu = np.ones((N_eofs, len(lat) * len(lon))) * -999.
    EOF_recons_Xv = np.ones((N_eofs, len(lat) * len(lon))) * -999.

    for i in range(N_eofs): 
        EOF_recons_Xu[i, sea] = EOFs_Xu[i, :]
        EOF_recons_Xv[i, sea] = EOFs_Xv[i, :]
        
    EOF_recons_Xu = ma.masked_values(np.reshape(EOF_recons_Xu, (N_eofs, len(lat), len(lon))), -999.)
    EOF_recons_Xv = ma.masked_values(np.reshape(EOF_recons_Xv, (N_eofs, len(lat), len(lon))), -999.)
    
    return {'EOFs_u' : EOFs_Xu, 'EOFs_v' : EOFs_Xv, 'EOFs_u_2d' : EOF_recons_Xu, 
            'EOFs_v_2d' : EOF_recons_Xv}



def eofs_2d_to_1d(EOFs, N_eofs):
    """
    Reshape EOFs from 2D to 1D, masking land points and taking a subset of EOFs
    """

    #Select subset of PCs
    EOFs_u = EOFs['u'][:N_eofs, :, :]
    EOFs_v = EOFs['v'][:N_eofs, :, :]

    #Extract mask and number of sea points
    sea = ~EOFs_u.mask
    sea_0 = ~EOFs_u[0, :, :].mask
    N_sea = (sea_0 == 1).sum()

    #reshape 2D to 1D - Concatenate results
    EOFs_v = np.reshape(EOFs_v[sea], (N_eofs, N_sea))
    EOFs_u = np.reshape(EOFs_u[sea], (N_eofs, N_sea))
    V_svd = pd.concat([pd.DataFrame(EOFs_u.T), pd.DataFrame(EOFs_v.T)], axis = 1)
    EOFs = np.vstack((EOFs_u, EOFs_v))

    return EOFs_u, EOFs_v, EOFs, V_svd

def reduced_data(df, EOFs_u, EOFs_v):
    n_input_points = int(df.shape[1]/2)
    Yu = pd.DataFrame(np.dot(df.iloc[:, :n_input_points], EOFs_u.T))
    Yv = pd.DataFrame(np.dot(df.iloc[:, n_input_points:], EOFs_v.T))
    df_output = pd.concat([Yu, Yv], axis = 1)

    return df_output