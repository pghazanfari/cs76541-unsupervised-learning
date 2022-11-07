import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import stats

def render_table(row_labels, col_labels, cell_data, table_scale=(1, 4), **kwargs):
    fig, ax = plt.subplots(**kwargs)
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(
        cellText=cell_data, 
        rowLabels=col_labels, 
        colLabels=row_labels, 
        cellLoc='center',
        loc='upper left')
    table.scale(*table_scale)
    fig.tight_layout()
    fig.show()
    return fig, ax

def closest_int(x):
    ceil = math.ceil(x)
    floor = math.floor(x)

    if ceil - x < x - floor:
        return int(ceil)
    else:
        return int(floor)

def predict_majority_classification(n_clusters, cluster_labels, y):
    cluster_preds = []
    for i in range(n_clusters):
        idx = np.argwhere(cluster_labels == i).flatten()
        if len(idx) == 0:
            cluster_preds.append(-1)
        else:
            cluster_preds.append(stats.mode(y[idx]).mode[0])

    assert len(cluster_preds) == n_clusters
    return np.take(cluster_preds, cluster_labels)

def predict_uniform_classification(n_clusters, cluster_labels, y):
    cluster_preds = []
    for i in range(n_clusters):
        idx = np.argwhere(cluster_labels == i).flatten()
        cluster_preds.append(closest_int(np.mean(y[idx])))
    assert len(cluster_preds) == n_clusters
    return np.take(cluster_preds, cluster_labels)

def predict_weighted_classification(n_clusters, cluster_labels, cluster_distances, y):
    cluster_distances = cluster_distances[np.arange(cluster_distances.shape[0]), cluster_labels]

    cluster_preds = []
    for i in range(n_clusters):
        idx = np.argwhere(cluster_labels == i).flatten()
        

        distances = 1.0 / (cluster_distances[idx] + .00001)
        avg = np.average(y[idx], weights=distances)

        cluster_preds.append(closest_int(avg))
    assert len(cluster_preds) == n_clusters
    return np.take(cluster_preds, cluster_labels)
