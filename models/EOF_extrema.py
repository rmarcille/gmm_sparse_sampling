# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 19:08:16 2023

@author: robin.marcille
"""

import numpy as np

def select_eofs_exrtrema(EOFs_u, EOFs_v, extrema):
    """Select and sort the indexes of the EOFs extrema of u and v, according to the extrema dictionnary

    Args:
        EOFs_u (numpy array): Zonal wind EOFs
        EOFs_v (numpy array): Meridional wind EOFs
        extrema (dict): Dictionnary containing the extrema to select for u and v, and EOFs from 1 to 5

    Returns:
        list: list of sorted indexes for the EOFs extrema sensors siting
    """
    n_eofs = [1, 2, 3, 4, 5]
    params = ['u','v']
    idx_extrema = []
    EOFs = {'u' : EOFs_u, 'v': EOFs_v}
    for i, n_eof in enumerate(n_eofs):
        size = []
        idxs = []
        for j, param in enumerate(params):            
            #Check if extrema for param and n_eof is to be selected
            ex_max = extrema[param][n_eof]['max'] == 1
            ex_min = extrema[param][n_eof]['min'] == 1

            #Get index of corresponding extrema
            idx_max = np.where(EOFs[param][n_eof, :] == np.max(EOFs[param][n_eof,:]))
            idx_min = np.where(EOFs[param][n_eof, :] == np.min(EOFs[param][n_eof,:]))

            #Get the associated magnitudes
            if ex_max : 
                size.append(abs(np.max(EOFs[param][n_eof,:])))
                idxs.append(idx_max)
            if ex_min : 
                size.append(abs(np.min(EOFs[param][n_eof,:])))            
                idxs.append(idx_min)

        #Sort the extrema for EOFs number n_eofs per magnitude
        arrinds = np.array(size).argsort()
        idxs = np.array(idxs)[arrinds[::-1]]

        #Append in final index list
        for idx in idxs : 
            idx_extrema.append(idx[0][0])
        
    return idx_extrema