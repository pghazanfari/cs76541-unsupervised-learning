from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import random_center_initializer
from pyclustering.cluster import cluster_visualizer_multidim

import utils

class PyclusterKMeansEstimator(BaseEstimator):
    def __init__(self, n_clusters=2, random_state=None, **kwargs):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kwargs = kwargs
        self._estimator_type = "clusterer"
        #getattr(estimator, "_estimator_type", None) == "clusterer"

    def set_params(self, **kwargs):
        self.n_clusters = kwargs.get('n_clusters', self.n_clusters)
        self.random_state = kwargs.get('random_state', self.random_state)

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        self.initial_centers = random_center_initializer(X, self.n_clusters).initialize()
        self.estimator = kmeans(X, self.initial_centers, **self.kwargs)
        self.estimator.process()
        # Fixing a bug in the library?
        self.estimator._kmeans__centers = np.array(self.estimator._kmeans__centers)
        self.labels_ = self.estimator.predict(X)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return self.estimator.predict(X)

    def fit_predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        self.fit(X)
        return self.predict(X)

    @property
    def cluster_centers(self):
        return self.estimator.get_centers()

    @property
    def inertia_(self):
        return self.estimator.get_total_wce()

    def visualize(self, X, **kwargs):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if X.shape[1] <= 3:
            kmeans_visualizer.show_clusters(X, 
                self.estimator.get_clusters(), 
                self.estimator.get_centers(), 
                self.initial_centers, **kwargs)
        else:
            visualizer = cluster_visualizer_multidim()
            visualizer.append_clusters(self.estimator.get_clusters(), X.tolist())
            visualizer.show(**kwargs)

class GaussianMixtureEstimator(GaussianMixture):
    def __init__(self, n_components=1, **kwargs):
        super().__init__(n_components=1, **kwargs)
        self._estimator_type = "clusterer"

    def set_params(self, **kwargs):
        if 'n_clusters' in kwargs:
            kwargs['n_components'] = kwargs['n_clusters']
            del kwargs['n_clusters']
        return super().set_params(**kwargs)

    def fit(self, X, y=None):
        super().fit(X, y)
        self.labels_ = self.predict(X)
