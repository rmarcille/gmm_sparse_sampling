# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 16:22:15 2023

@author: robin.marcille
"""

import numpy as np
import pandas as pd 

def wind_reconstruction_from_index(df_med, EOFs, gamma_index, train_index, test_index):
    """Reconstruct wind speed from a limited number of measurements

    Args:
        df_med (pandas DataFrame): Zonal and meridional wind speed in 2D (time, grid points)
        EOFs (numpy array): Empirical Orthogonal functions (Concatenated u and v) (n EOFs, grid points)
        gamma_index (numpy array): list of index to consider as input
        train_index (numpy array): time index of training dataset
        test_index (numpy array): time index of test dataset

    Returns:
        X_pred_test (numpy array) : Reconstructed predicted wind speed on test set
        X_true_test (numpy array) : True reduced wind speed on test set
        X_true_test (numpy array) : True reduced wind speed on train set
    """
    n_inputs_points = int(df_med.shape[1]/2)
    n_eofs = int(EOFs.shape[0]/2)
    
    #Coefficients in the reduced basis - to be predicted
    a_output_u = pd.DataFrame(np.dot(df_med.iloc[:, :n_inputs_points], EOFs[:n_eofs, :].T))
    a_output_v = pd.DataFrame(np.dot(df_med.iloc[:, n_inputs_points:], EOFs[n_eofs:, :].T))
    a_output = pd.concat((a_output_u, a_output_v), axis = 1)
    a_output.index = df_med.index
    
    #The observed wind speed - input
    Y_obs = df_med.iloc[:, gamma_index]

    #Train - test split
    Y_obs_test = Y_obs.iloc[test_index, :]
    Y_obs_train = Y_obs.iloc[train_index, :]

    a_true_train = a_output.iloc[train_index, :]
    a_true_test = a_output.iloc[test_index, :]

    #Perform linear regression between observed wind speed and reduced basis coefficients
    beta_hat = np.dot(np.dot(np.linalg.inv(np.dot(Y_obs_train.T, Y_obs_train)), Y_obs_train.T), a_true_train)

    #Compute reduced basis prediction from measurements on test split
    a_pred_test = np.dot(beta_hat.T,  Y_obs_test.T)

    #Reconstruct the full states
    X_pred_test_u = np.dot(EOFs[:n_eofs, :].T, a_pred_test[:n_eofs, :])
    X_pred_test_v = np.dot(EOFs[n_eofs:, :].T, a_pred_test[n_eofs:, :])
    X_pred_test = np.vstack((X_pred_test_u, X_pred_test_v)).T

    X_true_test_u = np.dot(EOFs[:n_eofs, :].T, a_true_test.iloc[:, :n_eofs].T)
    X_true_test_v = np.dot(EOFs[n_eofs:, :].T, a_true_test.iloc[:, n_eofs:].T)
    X_true_test = np.vstack((X_true_test_u, X_true_test_v)).T

    X_true_train_u = np.dot(EOFs[:n_eofs, :].T, a_true_train.iloc[:, :n_eofs].T)
    X_true_train_v = np.dot(EOFs[n_eofs:, :].T, a_true_train.iloc[:, n_eofs:].T)
    X_true_train = np.vstack((X_true_train_u, X_true_train_v)).T

    return X_pred_test, X_true_test, X_true_train
