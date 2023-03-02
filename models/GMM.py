# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 19:08:16 2023

@author: robin.marcille
"""
import scipy
from sklearn import mixture
import numpy as np


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