# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 19:08:16 2023

@author: robin.marcille
"""
import scipy
from sklearn import mixture
import numpy as np
from tqdm import tqdm
import pandas as pd
import random

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from scoring.scores import score_spatially_normalized
from scoring.reconstruction import wind_reconstruction_from_index


def fit_gmm_and_extract_centroids(EOFs, n_clusters, random_state = 10):
    """Fits a GMM model on the EOFs data and extracts centroids index

    Args:
        EOFs (numpy array): Input EOFs data dimension = (N_grid_points, N_eofs)
        n_clusters (int): Number of clusters to fit
        random_state (int, optional): _description_. Defaults to 10.

    Returns:
        idx_centroids (list): list of centroids index
    """
    X = EOFs.T
    gmm = mixture.GaussianMixture(n_components = n_clusters, covariance_type='full', random_state = random_state).fit(X)
    idx_centers_n_clusters_states = []
    centers = np.empty(shape=(gmm.n_components, X.shape[1]))
    for i in range(gmm.n_components):
        density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(X)
        centers[i, :] = X[np.argmax(density)]
    idx_centroids = [np.where(X[:, 0] == centers[i, 0])[0][0] for i in range(gmm.n_components)]
    return idx_centroids



def point_proportion_under_threshold(df_med, EOFs, thresholds, idx_train, idx_test, min_sens = 1, max_sens = 10, n_random_state = 10):
    """Compute the minimum number of input so that (percent)% of the grid points have a Normalized RMSE
        inferior to (threshold)

    Args:
        df_med (pandas DataFrame): wind speed DataFrame
        EOFs (numpy array): EOFs of zonal and meridional wind speed
        thresholds (float): List of N-RMSE threshold to consider
        idx_train (numpy array): Index of training set
        idx_test (numpy array): index of test set
        min_sens (int, optional): Minimun number of sensors to compute for. Defaults to 1.
        max_sens (int, optional): Maximun number of sensors to compute for. Defaults to 10.
        n_random_state (int, optional): Number of random state to compute for. Defaults to 10.

    Returns:
        percent_thresh_states (numpy array): array of percentages of maps under threshold, of shape (n_random_state, n_clusters, n_thresholds)
    """

    n_input_points = int(EOFs.shape[1])
    random_states = [random.randint(0, 100) for i in range(n_random_state)]

    percent_thresh_states = {} 
    list_clusters = {}
    for n in range(min_sens, max_sens + 1):
        list_clusters[n] = []
    for t in thresholds:
        percent_thresh_states[str(t)] = list_clusters

    percent_thresh_states = np.zeros((n_random_state, max_sens - min_sens + 1, len(thresholds)))
    #Loop over random states
    for s, state in enumerate(tqdm(random_states)):
        idx_centers_n_clusters_states = []
        #Generate clusters for the given random states
        for n in range(min_sens, max_sens + 1):
            #Fit GMM and extract centroids
            idx_centers = fit_gmm_and_extract_centroids(EOFs, n, random_state = state)

            #Compute normalized error
            idx_input2 = list(np.array(idx_centers) + n_input_points)
            idx_centers = idx_centers + idx_input2
            Xpred, X_test, X_train = wind_reconstruction_from_index(df_med, EOFs, idx_centers, idx_train, idx_test)
            umag_rmse_map = score_spatially_normalized(Xpred, X_test, X_train)
            umag_rmse_map = pd.DataFrame(umag_rmse_map)

            for t, thresh in enumerate(thresholds) : 
                percent_thresh = (umag_rmse_map < thresh).sum()/n_input_points
                percent_thresh_states[s, n-1, t] = percent_thresh.values[0]

    return percent_thresh_states

def opt_numb_sensors_threshold(df_med, EOFs, threshold, percent, idx_train, idx_test, min_sens = 1, max_sens = 10, n_random_state = 10):
    """Compute the minimum number of input so that (percent)% of the grid points have a Normalized RMSE
        inferior to (threshold)

    Args:
        df_med (pandas DataFrame): wind speed DataFrame
        EOFs (numpy array): EOFs of zonal and meridional wind speed
        threshold (float): N-RMSE threshold to consider
        percent (float): Proportion of the grid points to have under threshold
        idx_train (numpy array): Index of training set
        idx_test (numpy array): index of test set
        min_sens (int, optional): Minimun number of sensors to compute for. Defaults to 1.
        max_sens (int, optional): Maximun number of sensors to compute for. Defaults to 10.
        n_random_state (int, optional): Number of random state to compute for. Defaults to 10.

    Returns:
        n_sensors (int): Minimum number of sensors to reach the error threshold
    """

    percent_thresh_states = {}
    n_input_points = int(EOFs.shape[1])
    random_states = [random.randint(0, 100) for i in range(n_random_state)]

    #Loop over random states
    for s, state in enumerate(tqdm(random_states)):
        idx_centers_n_clusters_states = []
        #Generate clusters for the given random states
        for n in range(min_sens, max_sens + 1):
            #Fit GMM and extract centroids
            idx_centers = fit_gmm_and_extract_centroids(EOFs, n, random_state = state)

            #Compute normalized error
            idx_input2 = list(np.array(idx_centers) + n_input_points)
            idx_centers = idx_centers + idx_input2
            Xpred, X_test, X_train = wind_reconstruction_from_index(df_med, EOFs, idx_centers, idx_train, idx_test)
            umag_rmse_map = score_spatially_normalized(Xpred, X_test, X_train)
            umag_rmse_map = pd.DataFrame(umag_rmse_map)

            #Proportion of points under threshold
            percent_thresh = (umag_rmse_map < threshold).sum()/n_input_points
            if s == 0:
                percent_thresh_states[n] = [percent_thresh]
            else: 
                percent_thresh_states[n].append(percent_thresh)

    #Take the mean of the percentages
    mean_percent = []
    for n in range(min_sens, max_sens + 1):
        mean_percent.append(np.mean(percent_thresh_states[n]))
    
    #Compute minimum number of sensors to comply with the threshold
    n_sensors = np.argmax((np.array(mean_percent) > percent).cumsum() == 1) + 1

    return n_sensors