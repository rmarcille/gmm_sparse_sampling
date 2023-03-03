# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 16:22:15 2023

@author: robin.marcille
"""
import numpy as np
from sklearn.metrics import mean_squared_error

def score_reconstructed(X_pred, X_test):
    n_inputs = int(X_test.shape[1]/2)

    umagtest = np.sqrt(X_test[:, :n_inputs]**2 + X_test[: , n_inputs:]**2)
    umagpred = np.sqrt(X_pred[:, :n_inputs]**2 + X_pred[: , n_inputs:]**2)

    err_umag = mean_squared_error(umagtest, umagpred, squared = False)
    err_uv = mean_squared_error(X_pred, X_test, squared = False)

    max_umagpred = umagpred.max(axis = 0)
    max_umagtest = umagtest.max(axis = 0)
    err_umax = mean_squared_error(max_umagpred, max_umagtest, squared = False)

    mean_umagpred = umagpred.mean(axis = 0)
    mean_umagtest = umagtest.mean(axis = 0)
    err_umean = mean_squared_error(mean_umagpred, mean_umagtest, squared = False)

    return err_umag, err_uv, err_umax, err_umean

def score_spatially_normalized(X_pred, X_test):
    n_inputs = int(X_test.shape[1]/2)

    umagtest = np.sqrt(X_test[:, :n_inputs]**2 + X_test[: , n_inputs:]**2)
    umagpred = np.sqrt(X_pred[:, :n_inputs]**2 + X_pred[: , n_inputs:]**2)

    umagpred = (umagpred - umagtest.mean())/umagtest.std()
    umagtest = (umagtest - umagtest.mean())/umagtest.std()

    umag_map_nrmse = np.sqrt(((umagtest - umagpred)**2).mean(axis = 0))
    return umag_map_nrmse