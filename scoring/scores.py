# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 16:22:15 2023

@author: robin.marcille
"""
import numpy as np
from sklearn.metrics import mean_squared_error

def score_reconstructed(X_pred, X_test):
    """Score the wind speed reconstruction

    Args:
        X_pred (numpy array): Wind speed predictions 
        X_test (_type_): Reduced true wind speed on test set

    Returns:
        err_umag (float): RMSE on wind magnitude averaged over grid points and time
        err_uv (float): RMSE on wind components averaged over grid points, time and components
        err_umax (float): RMSE on maximum wind speed averaged over time
        err_umean (float): RMSE on mean wind speed averaged over time
    """
    #number of input points
    n_inputs = int(X_test.shape[1]/2)

    #Compute wind magnitude
    umagtest = np.sqrt(X_test[:, :n_inputs]**2 + X_test[: , n_inputs:]**2)
    umagpred = np.sqrt(X_pred[:, :n_inputs]**2 + X_pred[: , n_inputs:]**2)

    #Compute errors
    err_umag = mean_squared_error(umagtest, umagpred, squared = False)
    err_uv = mean_squared_error(X_pred, X_test, squared = False)

    #Compute errors on max and mean wind speed
    max_umagpred = umagpred.max(axis = 0)
    max_umagtest = umagtest.max(axis = 0)
    err_umax = mean_squared_error(max_umagpred, max_umagtest, squared = False)

    mean_umagpred = umagpred.mean(axis = 0)
    mean_umagtest = umagtest.mean(axis = 0)
    err_umean = mean_squared_error(mean_umagpred, mean_umagtest, squared = False)

    return err_umag, err_uv, err_umax, err_umean

def score_spatially_normalized(X_pred, X_test, X_train):
    """Compute the Normlaized RMSE

    Args:
        X_pred (numpy array): Wind speed predictions 
        X_test (numpy array): Reduced true wind speed on test set
        X_train (numpy array): Reduced true wind speed on training set

    Returns:
        umag_map_nrmse: RMSE on normalized wind magnitude averaged over time
    """
    #number of input points
    n_inputs = int(X_test.shape[1]/2)

    #Compute normalization parameters
    mean_u = np.mean(X_train[:, :n_inputs])
    mean_v = np.mean(X_train[:, n_inputs:])
    std_u = np.std(X_train[:, :n_inputs])
    std_v = np.std(X_train[:, n_inputs:])

    #Compute normalized wind magnitude
    umagtest = np.sqrt(((X_test[:, :n_inputs] - mean_u)/std_u)**2 + ((X_test[: , n_inputs:] - mean_u)/std_u)**2)
    umagpred = np.sqrt(((X_pred[:, :n_inputs]**2 - mean_v)/std_v)**2 + ((X_pred[: , n_inputs:] - mean_v)/std_v)**2)

    #Compute error averaged over time
    umag_map_nrmse = np.sqrt(((umagtest - umagpred)**2).mean(axis = 0))
    return umag_map_nrmse