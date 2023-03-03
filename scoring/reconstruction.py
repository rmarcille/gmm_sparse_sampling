# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 16:22:15 2023

@author: robin.marcille
"""

import numpy as np
import pandas as pd 

def wind_reconstruction_from_index(df_med, EOFs, gamma_index, train_index, test_index):

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

    #Reconstruct the full state
    X_pred_test_u = np.dot(EOFs[:n_eofs, :].T, a_pred_test[:n_eofs, :])
    X_pred_test_v = np.dot(EOFs[n_eofs:, :].T, a_pred_test[n_eofs:, :])
    X_pred_test = np.vstack((X_pred_test_u, X_pred_test_v)).T


    X_true_test_u = np.dot(EOFs[:n_eofs, :].T, a_true_test.iloc[:, :n_eofs].T)
    X_true_test_v = np.dot(EOFs[n_eofs:, :].T, a_true_test.iloc[:, n_eofs:].T)
    X_true_test = np.vstack((X_true_test_u, X_true_test_v)).T

    return X_pred_test, X_true_test

def LeastSquares_reconstruction(Y_obs_train, a_train, Y_obs_test): 
    Y_obs_train = np.hstack((np.ones((Y_obs_train.shape[0], 1)), Y_obs_train))
    beta_hat = np.dot(np.dot(np.linalg.inv(np.dot(Y_obs_train.T, Y_obs_train)), Y_obs_train.T),a_train)
    a_recons = np.dot(beta_hat.T, np.hstack((np.ones((Y_obs_test.shape[0], 1)), Y_obs_test)).T).T
    return a_recons

def inverse_reconstruction(Y_obs_test, C_gamma, V):
    return np.dot(np.linalg.pinv(np.dot(C_gamma, V)), Y_obs_test)
