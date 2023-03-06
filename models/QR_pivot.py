# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 19:08:16 2023

@author: robin.marcille
"""
import numpy as np
from scipy.linalg import qr as qr

def qr_pivots(EOFs, n_inputs):
    """Compute QR decomposition and extract pivot index

    Args:
        EOFs (numpy array): array of EOFs data
        n_inputs (int): number of pivots

    Returns:
        idx_input_qr (list): list of pivot index 
    """

    #Compute decomposition
    Q, R, pivot = qr(EOFs, pivoting=True)

    #Extract n_inputs first indices
    idx_input_qr = pivot[:n_inputs]
    
    return idx_input_qr