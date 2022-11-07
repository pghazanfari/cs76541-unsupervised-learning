from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.base import BaseEstimator

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import random_center_initializer
from pyclustering.cluster import cluster_visualizer_multidim

import utils

def elbow_plot_analysis(X, estimator, max_clusters=2, fig=None):
    assert max_clusters >= 2
    cluster_sizes = range(2, max_clusters+1)

    distortions = []
    for ci, n_clusters in enumerate(cluster_sizes):
        estimator.set_params(n_clusters=n_clusters)
        estimator.fit(X)
        distortions.append(estimator.inertia_)

    fig = fig or plt.figure()
    ax = fig.gca()
    ax.plot(cluster_sizes, distortions)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.show()

# Source: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#
def silhouette_score_analysis(X, estimator, max_clusters=2, ncols=2, **kwargs):
    assert max_clusters >= 2
    
    cluster_sizes = range(2, max_clusters+1)
    nrows = int(math.ceil(len(cluster_sizes) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, **kwargs)

    avg_scores = []
    for ci, n_clusters in enumerate(cluster_sizes):
        if ncols > 1:
            ax = axes[ci // ncols, ci % ncols]
        else:
            ax = axes[ci]

        ax.set_xlim([-0.1, 1])
        #ax.set_xlim([-0.1, 0.4])
        ax.set_ylim([0, len(X) + (n_clusters + 1) * (max_clusters+1)])


        estimator.set_params(n_clusters=n_clusters)
        clusters = estimator.fit_predict(X)
        silhouette_score_avg = silhouette_score(X, clusters)
        silhouette_scores = silhouette_samples(X, clusters)

        avg_scores.append([n_clusters, silhouette_score_avg])

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = silhouette_scores[clusters == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        # ax.set_title("The silhouette plot for the various clusters.")
        # ax.set_xlabel("The silhouette coefficient values")
        # ax.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_score_avg, color="red", linestyle="--")

        ax.set_yticks([])  # Clear the yaxis labels / ticks
        #ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4])

    if len(cluster_sizes) % ncols > 0:
        if ncols == 1:
            fig.delaxes(axes[-1])
        else:
            fig.delaxes(axes[-1, -1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.show()

    utils.render_table(
        ['Cluster Count', 'Avg Silhouette Score'], 
        ['' for _ in range(2, max_clusters+1)], 
        avg_scores,
        table_scale=(1, 2))