import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, learning_curve, validation_curve, train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics.cluster import homogeneity_score, completeness_score, adjusted_rand_score
from scipy.stats import kurtosis
from scipy.special import softmax
from scipy import stats
from pyclustering.utils.metric import distance_metric, type_metric

from yellowbrick.cluster import KElbowVisualizer

import utils
from common import *
from estimators import *
from projection import *
from scoring import *

class Experiment:
    def __init__(self, X, y, rng=104582):
        self.X = X
        self.y = y
        self.rng = rng
        self.state = {}
        self.metrics = {}

    def reset(self):
        self.state = {}

    def predict_majority_classification(self, n_clusters, cluster_labels, y):
        cluster_preds = []
        for i in range(n_clusters):
            idx = np.argwhere(cluster_labels == i).flatten()
            cluster_preds.append(stats.mode(y[idx]).mode[0])
        assert len(cluster_preds) == n_clusters
        return np.take(cluster_preds, cluster_labels)

    def predict_uniform_classification(self, n_clusters, cluster_labels, y):
        cluster_preds = []
        for i in range(n_clusters):
            idx = np.argwhere(cluster_labels == i).flatten()
            cluster_preds.append(utils.closest_int(np.mean(y[idx])))
        assert len(cluster_preds) == n_clusters
        return np.take(cluster_preds, cluster_labels)

    def predict_weighted_classification(self, n_clusters, cluster_labels, cluster_distances, y):
        cluster_distances = cluster_distances[np.arange(cluster_distances.shape[0]), cluster_labels]

        cluster_preds = []
        for i in range(n_clusters):
            idx = np.argwhere(cluster_labels == i).flatten()
            

            distances = 1.0 / (cluster_distances[idx] + .00001)
            avg = np.average(y[idx], weights=distances)

            cluster_preds.append(utils.closest_int(avg))
        assert len(cluster_preds) == n_clusters
        return np.take(cluster_preds, cluster_labels)

    def visualize_elbow(self, x, y, estimator, max_clusters, fig=None, axes=None, classification='majority'):
        if fig is None and axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(20,7))
        elif fig is None:
            fig = axes.get_figure()
        else:
            axes = fig.subplots(1, 2)

        visualizer = KElbowVisualizer(estimator, k=(2, max_clusters), fig=fig, ax=axes[0])
        visualizer.fit(x)
        
        scores = {
            'Homogeneity': {
                'fn': homogeneity_score
            },
        }

        for k in scores:
            scores[k]['values'] = []

        for n_clusters in range(2, max_clusters+1):
            estimator.set_params(n_clusters=n_clusters)
            
            if classification == 'majority':
                y_pred = self.predict_majority_classification(n_clusters, estimator.fit_predict(x), y)
            elif classification == 'uniform':
                y_pred = self.predict_uniform_classification(n_clusters, estimator.fit_predict(x), y)
            elif classification == 'weighted':
                y_pred = self.predict_weighted_classification(n_clusters, estimator.fit_predict(x), estimator.transform(x), y)
            else:
                raise NotImplementedError
            
            for key in scores:
                scores[key]['values'].append(scores[key]['fn'](y, y_pred))
        
        for key in scores:
            axes[1].plot(scores[key]['values'], label=key)

        axes[1].set_xlabel("n clusters")
        axes[1].set_ylabel("Score")
        axes[1].legend()

        visualizer.show()

    def visualize_silhouette(self, **kwargs):
        self.state = {**self.state, **kwargs}
        #estimator = self.state.get('estimator', PyclusterKMeansEstimator(random_state=self.rng, metric=distance_metric(type_metric.EUCLIDEAN)))
        estimator = self.state.get('estimator', KMeans(random_state=self.rng))
        max_clusters = self.state.get('max_clusters', 20)
        silhouette_score_analysis(self.X, estimator, max_clusters=max_clusters, ncols=5, figsize=(20,20))
        self.state['estimator'] = estimator
        self.state['max_clusters'] = max_clusters

    def perform_pca(self, **kwargs):
        self.state = {**self.state, **kwargs}
        n_components = self.state.get('pca', {}).get('n_components', 0.95)

        pca = PCA(n_components=n_components)
        pca_X = pca.fit_transform(self.X)

        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        if n_components > 0 and n_components < 1:
            plt.gca().hlines(y=n_components, xmin=0, xmax=len(pca.explained_variance_ratio_), linewidth=2, color='r', linestyles='dashed')
        plt.xlabel('Number of PCA Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.show()

        rows = range(1, len(pca.explained_variance_ratio_)+1)
        cols = ['Explained Variance Ratio (EVR)', 'Cumulative EVR']
        cell_data = np.array([pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)]).T
        #cell_data = np.array([pca.explained_variance_ratio_]).T
        utils.render_table(cols, rows, cell_data, table_scale=(1,4), figsize=(10,20))

        self.state['pca'] = self.state.get('pca', {})
        self.state['pca']['PCA'] = pca
        self.state['pca']['pca_X'] = pca_X
        self.state['pca']['n_components'] = len(pca.explained_variance_ratio_)

    def perform_ica(self, **kwargs):
        self.state = {**self.state, **kwargs}

        n_components = self.state.get('ica', {}).get('n_components', self.X.shape[1])
        ica = FastICA(n_components, random_state=self.rng, whiten='unit-variance')
        ica_X = ica.fit_transform(self.X)
        ica_kurtosis = kurtosis(ica_X, axis=0)
        self.state['ica'] = self.state.get('ica', {})
        self.state['ica']['ica_X'] = ica_X
        
        plt.figure()
        plt.title("Kurtosis values per component")
        plt.ylabel("Kurtosis")
        plt.xlabel("Component")
        plt.bar(range(1, ica_kurtosis.shape[0]+1), ica_kurtosis, label=range(1, ica_kurtosis.shape[0]+1))
        plt.show()

    def perform_random_projection(self, **kwargs):
        self.state = {**self.state, **kwargs}

        n = self.state.get('grp', {}).get('n', 10)
        self.state['grp'] = self.state.get('grp', {})
        self.state['grp']['grp_X'] = []
        avg_loss = []
        for n_components in range(1, self.X.shape[1]+1):
            loss = []
            for i in range(n):
                grp = GaussianRandomProjection(n_components, random_state=self.rng+i)
                grp_X = grp.fit_transform(self.X)
                y_rec = grp.inverse_transform(grp_X)
                loss.append(mean_squared_error(y_rec, self.X))
            avg_loss.append(np.mean(loss))

        
        plt.figure()
        plt.title("Gaussian Random Project Reconstruction Loss vs Number of Components")
        plt.xlabel("Number of Components")
        plt.ylabel("Reconstruction Loss")
        plt.plot(range(1, len(avg_loss)+1), avg_loss)
        plt.show()

    def perform_custom_feature_transformation(self, **kwargs):
        self.state = {**self.state, **kwargs}
        svc = SVC(kernel='linear', C=0.1057371263440564)
        svc.fit(self.X, self.y)
        model = SelectFromModel(svc, prefit=True)
        X_new = model.transform(self.X)
        print(f"before={self.X.shape}, after={X_new.shape}")
        self.state['cft'] = self.state.get('cft', {})
        self.state['cft']['cft_X'] = X_new

    def perform_pca_kmeans(self, **kwargs):
        self.state = {**self.state, **kwargs}
        x = self.state.get('pca').get('pca_X')
        estimator = self.state.get('kmeans', {}).get('estimator', KMeans(random_state=self.rng))
        #estimator = KMeans(random_state=self.rng)
        max_clusters = self.state.get('max_clusters', 20)
        fig = plt.figure(figsize=(20,7))
        fig.suptitle("PCA Clustering (KMeans)")
        self.visualize_elbow(x, self.y, estimator, max_clusters, fig=fig)

    def perform_pca_em(self, **kwargs):
        self.state = {**self.state, **kwargs}
        x = self.state.get('pca').get('pca_X')
        estimator = self.state.get('em', {}).get('estimator', GaussianMixtureEstimator(random_state=self.rng))
        #estimator = GaussianMixtureEstimator(random_state=self.rng)
        max_clusters = self.state.get('max_clusters', 20)
        fig = plt.figure(figsize=(20,7))
        fig.suptitle("PCA Clustering (EM)")
        self.visualize_elbow(x, self.y, estimator, max_clusters, fig=fig)

    def perform_ica_kmeans(self, **kwargs):
        self.state = {**self.state, **kwargs}
        x = self.state.get('ica').get('ica_X')
        estimator = self.state.get('kmeans', {}).get('estimator', KMeans(random_state=self.rng))
        #estimator = KMeans(random_state=self.rng)
        max_clusters = self.state.get('max_clusters', 20)
        fig = plt.figure(figsize=(20,7))
        fig.suptitle("ICA Clustering (KMeans)")
        self.visualize_elbow(x, self.y, estimator, max_clusters, fig=fig)

    def perform_ica_em(self, **kwargs):
        self.state = {**self.state, **kwargs}
        x = self.state.get('ica').get('ica_X')
        estimator = self.state.get('em', {}).get('estimator', GaussianMixtureEstimator(random_state=self.rng))
        #estimator = GaussianMixtureEstimator(random_state=self.rng)
        max_clusters = self.state.get('max_clusters', 20)
        fig = plt.figure(figsize=(20,7))
        fig.suptitle("ICA Clustering (EM)")
        self.visualize_elbow(x, self.y, estimator, max_clusters, fig=fig)

    def perform_random_projection_kmeans(self, **kwargs):
        self.state = {**self.state, **kwargs}

        n_components = self.state.get('grp', {}).get('n_components', self.X.shape[1])
        n = self.state.get('grp', {}).get('n', 10)

        grp = GaussianRandomProjection(n_components, random_state=self.rng)

        # TODO: Average?
        x = grp.fit_transform(self.X)
        estimator = self.state.get('kmeans', {}).get('estimator', KMeans(random_state=self.rng))
        max_clusters = self.state.get('max_clusters', 20)
        fig = plt.figure(figsize=(20,7))
        fig.suptitle("Random Projection Clustering (KMeans)")
        self.visualize_elbow(x, self.y, estimator, max_clusters, fig=fig)

    def perform_random_projection_em(self, **kwargs):
        self.state = {**self.state, **kwargs}

        n_components = self.state.get('grp', {}).get('n_components', self.X.shape[1])
        n = self.state.get('grp', {}).get('n', 10)

        grp = GaussianRandomProjection(n_components, random_state=self.rng)

        # TODO: Average?
        x = grp.fit_transform(self.X)
        estimator = self.state.get('em', {}).get('estimator', GaussianMixtureEstimator(random_state=self.rng))
        max_clusters = self.state.get('max_clusters', 20)
        fig = plt.figure(figsize=(20,7))
        fig.suptitle("Random Projection Clustering (EM)")
        self.visualize_elbow(x, self.y, estimator, max_clusters, fig=fig)

    def perform_cft_kmeans(self, **kwargs):
        self.state = {**self.state, **kwargs}
        x = self.state.get('cft').get('cft_X')
        estimator = self.state.get('kmeans', {}).get('estimator', KMeans(random_state=self.rng))
        max_clusters = self.state.get('max_clusters', 20)
        fig = plt.figure(figsize=(20,7))
        fig.suptitle("CFT Clustering (EM)")
        self.visualize_elbow(x, self.y, estimator, max_clusters, fig=fig)

    def perform_cft_em(self, **kwargs):
        self.state = {**self.state, **kwargs}
        x = self.state.get('cft').get('cft_X')
        estimator = self.state.get('em', {}).get('estimator', GaussianMixtureEstimator(random_state=self.rng))
        max_clusters = self.state.get('max_clusters', 20)
        fig = plt.figure(figsize=(20,7))
        fig.suptitle("CFT Clustering (EM)")
        self.visualize_elbow(x, self.y, estimator, max_clusters, fig=fig)

    def train_nn(self, x, y, mlp=None, title=None):
        cross_val = StratifiedKFold(shuffle=True, random_state=self.rng)

        if mlp is None:
            mlp = MLPClassifier(activation='relu', hidden_layer_sizes=(12), alpha=1.0, max_iter=1000)

        clfs = [clone(mlp) for _ in range(cross_val.get_n_splits(x, y))]

        metrics = {
            'log_loss': {
                'fn': log_loss
            },
            'npv_precision_avg': {
                'fn': npv_precision_avg
            }
        }
        for k in metrics:
            metrics[k]['train'] = []
            metrics[k]['test'] = []

        train_losses = []
        test_losses = []
        for i, train_test_indices in enumerate(cross_val.split(x, y)):
            train_indices, test_indices = train_test_indices
            X_train = x[train_indices]
            y_train = self.y.iloc[train_indices]
            
            X_test = x[test_indices]
            y_test = self.y.iloc[test_indices]
            
            for k in metrics:
                metrics[k]['train'].append([])
                metrics[k]['test'].append([])

            for epoch in range(1000):
                clfs[i].partial_fit(X_train, y_train, classes=[0, 1])
                for k in metrics:
                    metrics[k]['train'][-1].append(metrics[k]['fn'](y_train, clfs[i].predict(X_train)))
                    metrics[k]['test'][-1].append(metrics[k]['fn'](y_test, clfs[i].predict(X_test)))

        for k in metrics:
            metrics[k]['train'] = np.mean(metrics[k]['train'], axis=0)
            metrics[k]['test'] = np.mean(metrics[k]['test'], axis=0)


        fig, axes = plt.subplots(1, 2, figsize=(20,7))
        if title:
            fig.suptitle(title)
        axes[0].plot(metrics['log_loss']['train'], label='Training Loss')
        axes[0].plot(metrics['log_loss']['test'], label='Validation Loss')
        axes[0].set_ylabel('Log Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].legend()

        axes[1].plot(metrics['npv_precision_avg']['train'], label='Training Loss')
        axes[1].plot(metrics['npv_precision_avg']['test'], label='Validation Loss')
        axes[1].set_ylabel('NPV Precision Avg')
        axes[1].set_xlabel('Epochs')
        axes[1].legend()

        fig.show()

        return metrics, fig, axes

    def perform_nn_pca(self, **kwargs):
        self.state = {**self.state, **kwargs}
        self.train_nn(self.state.get('pca').get('pca_X'), self.y, title='PCA NN Training')
        
    def perform_nn_ica(self, **kwargs):
        self.state = {**self.state, **kwargs}
        self.train_nn(self.state.get('ica').get('ica_X'), self.y, title='ICA NN Training')

    def perform_nn_rp(self, **kwargs):
        self.state = {**self.state, **kwargs}
        n_components = self.state.get('grp', {}).get('n_components', 5)
        grp = GaussianRandomProjection(n_components, random_state=self.rng)
        self.train_nn(grp.fit_transform(self.X), self.y, title='Random Projection NN Training')

    def perform_nn_cft(self, **kwargs):
        self.state = {**self.state, **kwargs}
        self.train_nn(self.state.get('cft').get('cft_X'), self.y, title='Custom Feature Transformation NN Training')

    def perform_nn_kmeans(self, **kwargs):
        self.state = {**self.state, **kwargs}
        n_clusters = self.state.get('kmeans', {}).get('n_clusters', 2)
        estimator = KMeans(n_clusters, random_state=self.rng)
        labels = estimator.fit_predict(self.X)
        if np.unique(labels).shape[0] == 1:
            enc = OneHotEncoder()
        else:
            enc = OneHotEncoder(drop='first')
        one_hot = enc.fit_transform(labels.reshape(-1, 1)).toarray()
        one_hot = StandardScaler().fit_transform(one_hot)
        x = np.hstack((self.X.to_numpy(), one_hot))
        self.train_nn(x, self.y, title='KMeans NN Training')

    def perform_nn_em(self, **kwargs):
        self.state = {**self.state, **kwargs}
        n_clusters = self.state.get('em', {}).get('n_clusters', 2)
        estimator = GaussianMixtureEstimator(n_clusters, random_state=self.rng)
        labels = estimator.fit_predict(self.X)
        if np.unique(labels).shape[0] == 1:
            enc = OneHotEncoder()
        else:
            enc = OneHotEncoder(drop='first')
        one_hot = enc.fit_transform(labels.reshape(-1, 1)).toarray()
        one_hot = StandardScaler().fit_transform(one_hot)
        x = np.hstack((self.X.to_numpy(), one_hot))
        self.train_nn(x, self.y, title='EM NN Training')

    def cell_1(self, **kwargs):
        self.state = {**self.state, **kwargs}
        estimator = self.state.get('kmeans', {}).get('estimator', KMeans(random_state=self.rng))
        max_clusters = self.state.get('max_clusters', 20)
        self.visualize_elbow(self.X, self.y, estimator, max_clusters)
        self.state['kmeans'] = self.state.get('kmeans', {})
        self.state['kmeans']['estimator'] = estimator
        self.state['max_clusters'] = max_clusters

    def cell_2(self, **kwargs):
        #self.visualize_elbow(estimator=GaussianMixtureEstimator(random_state=self.rng))

        self.state = {**self.state, **kwargs}
        estimator = self.state.get('em', {}).get('estimator', GaussianMixtureEstimator(random_state=self.rng))
        max_clusters = self.state.get('max_clusters', 20)
        self.visualize_elbow(self.X, self.y, estimator, max_clusters)
        self.state['em'] = self.state.get('em', {})
        self.state['em']['estimator'] = estimator
        self.state['max_clusters'] = max_clusters

    cell_3 = perform_pca
    cell_4 = perform_ica
    cell_5 = perform_random_projection
    cell_6 = perform_custom_feature_transformation
    cell_7 = perform_pca_kmeans
    cell_8 = perform_pca_em
    cell_9 = perform_ica_kmeans
    cell_10 = perform_ica_em
    cell_11 = perform_random_projection_kmeans
    cell_12 = perform_random_projection_em
    cell_13 = perform_cft_kmeans
    cell_14 = perform_cft_em
    cell_15 = perform_nn_pca
    cell_16 = perform_nn_ica
    cell_17 = perform_nn_rp
    cell_18 = perform_nn_cft
    cell_19 = perform_nn_kmeans
    cell_20 = perform_nn_em
