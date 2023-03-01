# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 19:08:16 2023

@author: robin.marcille
"""

from scipy.linalg import qr as qr

def qr_pivots(EOFs, n_inputs):
    Q, R, pivot = qr(EOFs, pivoting=True)
    idx_input_qr = pivot[:n_inputs]