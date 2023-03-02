# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 16:22:15 2023

@author: robin.marcille
"""

import numpy as np

# def LeastSquares_reconstruction(X, Y, Xt, Npc=10): 
#     """Perform Least Squares regression to reconstruct the wind speed 

#     Args:
#         X (_type_): _description_
#         Y (_type_): _description_
#         Xt (_type_): _description_
#         Yt (_type_): _description_
#         V (_type_): _description_
#         Npc (int, optional): _description_. Defaults to 10.
#         nrmse (bool, optional): _description_. Defaults to False.

#     Returns:
#         _type_: _description_
#     """

#     X = np.hstack((np.ones((X.shape[0], 1)), X))
#     beta_hat = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T),Y)
#     Ypred = np.dot(beta_hat.T, np.hstack((np.ones((Xt.shape[0], 1)), Xt)).T).T

#     return Ypred


def LeastSquares_reconstruction(Y_obs_train, a_train, Y_obs_test): 
    Y_obs_train = np.hstack((np.ones((Y_obs_train.shape[0], 1)), Y_obs_train))
    beta_hat = np.dot(np.dot(np.linalg.inv(np.dot(Y_obs_train.T, Y_obs_train)), Y_obs_train.T),a_train)
    a_recons = np.dot(beta_hat.T, np.hstack((np.ones((Y_obs_test.shape[0], 1)), Y_obs_test)).T).T
    return a_recons

def inverse_reconstruction(Y_obs_test, C_gamma, V):
    return np.dot(np.linalg.pinv(np.dot(C_gamma, V)), Y_obs_test)
