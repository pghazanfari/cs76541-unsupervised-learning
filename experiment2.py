import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import time

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
from sklearn.model_selection import GridSearchCV
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
        self.reset()

    def reset(self):
        self.state = [{'X': self.X, 'y': self.y} for _ in range(5)]
        self.metrics = [{'X': self.X, 'y': self.y} for _ in range(5)]

    def run_step(self, step, render=True, update_state=True, **kwargs):
        assert step >= 0 and step < len(Experiment.step_fns)
        if update_state:
            self.state[step] = {**self.state[step], **kwargs}
        state = self.state[step]
        metrics = self.metrics[step]
        X = state['X']
        y = state['y']
        Experiment.step_fns[step](self, state, metrics, X, y)
        if render:
            Experiment.render_fns[step](self, state, metrics, X, y)

    def run(self, render=True, update_state=True, **kwargs):
        for i in range(len(Experiment.step_fns)):
            self.run_step(i, render=render, update_state=update_state, **kwargs)

    def step1(self, state, metrics, X, y):
        state['kmeans'] = state.get('kmeans', {})
        state['em'] = state.get('em', {})
        max_clusters = state.get('max_clusters', 20)
        state['max_clusters'] = max_clusters

        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        state['fig'] = fig
        state['axes'] = axes

        state['kmeans']['estimator'] = KMeans(random_state=self.rng)
        state['em']['estimator'] = GaussianMixtureEstimator(random_state=self.rng)

        metrics['kmeans'] = {}
        metrics['em'] = {}

        scores = {
            'homogeneity_score': homogeneity_score
        }

        for score in scores:
            metrics['kmeans'][score] = []
            metrics['em'][score] = []

        for i, key in enumerate(['kmeans', 'em']):
            estimator = state[key]['estimator']
            visualizer = KElbowVisualizer(estimator, k=(2, max_clusters), fig=fig, ax=axes[i, 0])
            visualizer.fit(X)
            state[key]['visualizer'] = visualizer
            state[key]['n_clusters'] = state[key].get('n_clusters', visualizer.elbow_value_)

            for score in scores:
                for n_clusters in range(2, max_clusters+1):
                    estimator.set_params(n_clusters=n_clusters)
                    y_pred = utils.predict_majority_classification(n_clusters, estimator.fit_predict(X), y)
                    metrics[key][score].append(scores[score](y, y_pred))

    def render_step1(self, state, metrics, X, y):
        fig = state['fig']
        axes = state['axes']
        for i, key in enumerate(['kmeans', 'em']):
            for score in metrics[key]:
                axes[i, 1].plot(metrics[key][score], label=score)
            axes[i, 1].set_xlabel('n_clusters')
            axes[i, 0].set_xlabel('n_clusters')
            axes[i, 0].set_ylabel("distortion score")
            axes[i, 1].legend()
        
        axes[0, 0].set_title('Distortion Score Elbow for KMeans Clustering')
        axes[1, 0].set_title('Distortion Score Elbow for EM Clustring')
        fig.show()

    def step2(self, state, metrics, X, y):
        methods = ['pca', 'ica', 'grp', 'svc']

        for method in methods:
            state[method] = state.get(method, {})
            metrics[method] = {}

        # PCA
        #n_components = state['pca'].get('n_components', 0.95)
        #state['pca']['n_components'] = n_components
        pca = PCA()
        pca_X = pca.fit_transform(X)
        state['pca']['model'] = pca 
        state['pca']['n_components'] = state['pca'].get('n_components', pca_X.shape[1])
        state['pca']['X'] = pca_X[:, :(state['pca']['n_components'])]
        metrics['pca']['explained_variance_ratio'] = pca.explained_variance_ratio_

        # ICA
        n_components = state['ica'].get('n_components', X.shape[1])
        state['ica']['n_components'] = n_components
        ica = FastICA(n_components, random_state=self.rng, whiten='unit-variance')
        ica_X = ica.fit_transform(X)
        ica_kurtosis = kurtosis(ica_X, axis=0)
        state['ica']['model'] = ica
        state['ica']['X'] = ica_X
        metrics['ica']['kurtosis'] = ica_kurtosis

        # GRP
        n = state['grp'].get('n', 10)
        avg_loss = []
        for n_components in range(1, X.shape[1]+1):
            loss = []
            for i in range(n):
                grp = GaussianRandomProjection(n_components, random_state=self.rng+i)
                grp_X = grp.fit_transform(X)
                y_rec = grp.inverse_transform(grp_X)
                loss.append(mean_squared_error(y_rec, X))
            avg_loss.append(np.mean(loss))
        metrics['grp']['avg_mse'] = avg_loss

        # SVC
        def npv_precision_avg_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            return npv_precision_avg(y, y_pred)

        if 'C' in state['svc']:
            svc = SVC(kernel='linear', C=state['svc']['C'])
            svc.fit(X, y)
        else:
            print("Performing grid search to discover best C...")
            gs = GridSearchCV(
                SVC(kernel='linear'), 
                {'C': np.logspace(np.log10(0.001), np.log10(1), num=10)}, 
                scoring=npv_precision_avg_scorer, 
                n_jobs=16, 
                cv=StratifiedKFold(shuffle=True, random_state=self.rng),
                refit=True)

            gs.fit(X, y)
            svc = gs.best_estimator_

        state['svc']['estimator'] = svc
        metrics['svc']['avg_mse'] = []
        metrics['svc']['threshold'] = []
        metrics['svc']['nfeatures'] = []

        best_threshold = None
        min_rloss = None

        state['svc']['thresholds'] = []
        state['svc']['models'] = []
        state['svc']['Xs'] = []
        thresholds = np.linspace(np.min(svc.coef_- .0001), np.max(svc.coef_), num=40)
        print("thresholds=", thresholds)
        for threshold in thresholds:
            model = SelectFromModel(svc, threshold=threshold, prefit=True)
            svc_X = model.transform(X)
            y_rec = model.inverse_transform(svc_X)
            rloss = mean_squared_error(y_rec, X)
            print(f"threshold={threshold}, rloss={rloss}")
            metrics['svc']['avg_mse'].append(rloss)
            metrics['svc']['threshold'].append(threshold)
            metrics['svc']['nfeatures'].append(svc_X.shape[1])
            state['svc']['thresholds'].append(threshold)
            state['svc']['models'].append(model)
            state['svc']['Xs'].append(svc_X)
            if min_rloss is None or rloss < min_rloss:
                min_rloss = rloss
                best_threshold = threshold

        state['svc']['threshold'] = state['svc'].get('threshold', best_threshold)

    def render_step2(self, state, metrics, X, y):
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))

        axes[0, 0].set_title("PCA: Cumulative Explained Variance Ratio")
        evr = metrics['pca']['explained_variance_ratio'].flatten()
        axes[0, 0].plot(np.insert(np.cumsum(evr), 0, 0))        
        if state['pca']['n_components'] < 1:
            axes[0, 0].hlines(y=state['pca']['n_components'], xmin=0, xmax=len(evr), linewidth=2, color='r', linestyles='dashed')
        else:
            axes[0, 0].axvline(x=state['pca']['n_components'], ymin=0, ymax=1, linewidth=2, color='r', linestyle='--')
        axes[0, 0].bar(range(1, len(evr)+1), evr)
        axes[0, 0].set_xlabel('Number of PCA Components')
        axes[0, 0].set_ylabel('Cumulative Explained Variance Ratio')

        axes[0, 1].set_title("ICA: Kurtosis values per component")
        axes[0, 1].set_ylabel("Kurtosis")
        axes[0, 1].set_xlabel("Component")
        axes[0, 1].bar(range(1, metrics['ica']['kurtosis'].shape[0]+1), metrics['ica']['kurtosis'], label=range(1, metrics['ica']['kurtosis'].shape[0]+1))

        axes[1, 0].set_title("Gaussian Random Project: Reconstruction Loss vs Number of Components")
        axes[1, 0].set_xlabel("Number of Components")
        axes[1, 0].set_ylabel("Reconstruction Loss")
        axes[1, 0].plot(range(1, len(metrics['grp']['avg_mse'])+1), metrics['grp']['avg_mse'])
        
        axes[1, 1].set_title("SVC: Reconstruction Loss vs Number of Components")
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].plot(metrics['svc']['threshold'], metrics['svc']['avg_mse'], marker='o', label='Reconstruction Loss',)
        ax = axes[1, 1].twinx()
        ax.plot(metrics['svc']['threshold'], metrics['svc']['nfeatures'], label='N Components', color='r')
        ax.legend()
        axes[1, 1].legend()
        axes[1, 1].axvline(x=state['svc']['threshold'], ymin=0, ymax=1.0, color='k', marker='o', linestyle='--', linewidth=2)
        fig.show()

    def step3(self, state, metrics, X, y):
        methods = ['pca', 'ica', 'grp', 'svc']
        for method in methods:
            state[method] = {**self.state[1].get(method, {}), **state.get(method, {})}
            metrics[method] = {}

        state['ica']['features'] = state['ica'].get('features', tuple(range(state['ica']['X'].shape[1])))
        state['ica']['X'] = state['ica']['X'][:, state['ica']['features']]

        state['grp']['n_components'] = state['grp'].get('n_components', X.shape[1])
        grp = GaussianRandomProjection(state['grp']['n_components'], random_state=self.rng)
        state['grp']['X'] = grp.fit_transform(X)

        state['svc']['threshold'] = state['svc'].get('threshold', state['svc'].get('thresholds', [0])[0])
        if state['svc']['threshold'] in state['svc']['thresholds']:
            idx = state['svc']['thresholds'].index(state['svc']['threshold'])
            assert idx >= 0
            state['svc']['model'] = state['svc']['models'][idx]
            state['svc']['X'] = state['svc']['Xs'][idx]
        else:
            model = SelectFromModel(state['svc']['estimator'], threshold=state['svc']['threshold'], prefit=True)
            state['svc']['X'] = model.fit_transform(X)
            state['svc']['model'] = model

        for method in methods:
            mstate = {'X': state[method]['X'], 'y': y}
            mmetrics = {'X': state[method]['X'], 'y': y}
            self.step1(mstate, mmetrics, state[method]['X'], y)
            state[method]['step1'] = mstate
            metrics[method]['step1'] = mmetrics


    def render_step3(self, state, metrics, X, y):
        methods = ['pca', 'ica', 'grp', 'svc']
        for method in methods:
            state[method]['step1']['fig'].suptitle(method)
            self.render_step1(state[method]['step1'], metrics[method]['step1'], state[method]['X'], y)

    def train_nn(self, x, y, mlp=None, n_splits=5, epochs=1000):
        cross_val = StratifiedKFold(shuffle=True, random_state=self.rng, n_splits=n_splits)

        if mlp is None:
            mlp = MLPClassifier(activation='relu', hidden_layer_sizes=(12), alpha=1.0, max_iter=epochs)

        clfs = [clone(mlp) for _ in range(cross_val.get_n_splits(x, y))]

        times = []
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
            y_train = y.iloc[train_indices]
            
            X_test = x[test_indices]
            y_test = y.iloc[test_indices]
            
            for k in metrics:
                metrics[k]['train'].append([])
                metrics[k]['test'].append([])

            st = time.time()
            for epoch in range(epochs):
                clfs[i].partial_fit(X_train, y_train, classes=[0, 1])
                for k in metrics:
                    metrics[k]['train'][-1].append(metrics[k]['fn'](y_train, clfs[i].predict(X_train)))
                    metrics[k]['test'][-1].append(metrics[k]['fn'](y_test, clfs[i].predict(X_test)))
            et = time.time()
            times.append(et - st)

        for k in metrics:
            metrics[k]['train'] = np.mean(metrics[k]['train'], axis=0)
            metrics[k]['test'] = np.mean(metrics[k]['test'], axis=0)


        return { 'scores': metrics, 'time': np.mean(times) }

    def step4(self, state, metrics, X, y):
        methods = ['pca', 'ica', 'grp', 'svc']
        for method in methods:
            state[method] = {**self.state[2].get(method, {}), **state.get(method, {})}
            metrics[method] = {}

        for method in methods:
            metrics[method] = self.train_nn(state[method]['X'], y, mlp=state[method].get('mlp'), epochs=state.get('epochs', 1000))


    def render_step4(self, state, metrics, X, y):
        methods = ['pca', 'ica', 'grp', 'svc']

        fig, axes = plt.subplots(2, 2, figsize=(20,14), sharex='all')
        fig.suptitle("Dim Reduction NN Training")
        axes[0, 0].set_title("Training Loss")
        axes[0, 1].set_title("Validation Loss")
        axes[1, 0].set_title("Training npv_precision_avg")
        axes[1, 1].set_title("Validation npv_precision_avg")

        for method in methods:
            axes[0, 0].plot(metrics[method]['scores']['log_loss']['train'], label=method)
            axes[0, 1].plot(metrics[method]['scores']['log_loss']['test'], label=method)
            axes[1, 0].plot(metrics[method]['scores']['npv_precision_avg']['train'], label=method)
            axes[1, 1].plot(metrics[method]['scores']['npv_precision_avg']['test'], label=method)

        perms = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for i, j in perms:
            axes[i, j].legend()
            axes[i, j].set_xlabel("Epochs")

        axes[1, 0].set_ylim(top=1.0)
        axes[1, 1].set_ylim(top=1.0)

        fig.set_tight_layout(True)
        fig.show()

        fig = plt.figure()
        fig.suptitle("NN Training Time (s)")
        times = [metrics[method]['time'] for method in methods]
        plt.bar(methods, times)
        fig.show()

    def step5(self, state, metrics, X, y):
        methods = ['kmeans', 'em']
        for method in methods:
            state[method] = {**self.state[0].get(method, {}), **state.get(method, {})}
            state[method] = {**self.state[2].get(method, {}), **state.get(method, {})}
            metrics[method] = {}

        estimators = {
            'kmeans': KMeans(state['kmeans']['n_clusters'], random_state=self.rng),
            'em': GaussianMixture(state['em']['n_clusters'], random_state=self.rng)
        }

        for method in methods:
            estimator = estimators[method]
            labels = estimator.fit_predict(X)
            if np.unique(labels).shape[0] == 1:
                enc = OneHotEncoder()
            else:
                enc = OneHotEncoder(drop='first')
            one_hot = enc.fit_transform(labels.reshape(-1, 1)).toarray()
            one_hot = StandardScaler().fit_transform(one_hot)
            x = np.hstack((X.to_numpy(), one_hot))
            state[method]['X'] = x
            metrics[method] = self.train_nn(x, y, epochs=state.get('epochs', 1000))

    def render_step5(self, state, metrics, X, y):
        methods = ['kmeans', 'em']

        fig, axes = plt.subplots(2, 2, figsize=(20,14), sharex='all')

        fig.suptitle("Cluster Feature NN Training")
        axes[0, 0].set_title("Training Loss")
        axes[0, 1].set_title("Validation Loss")
        axes[1, 0].set_title("Training npv_precision_avg")
        axes[1, 1].set_title("Validation npv_precision_avg")

        for method in methods:
            axes[0, 0].plot(metrics[method]['scores']['log_loss']['train'], label=method)
            axes[0, 1].plot(metrics[method]['scores']['log_loss']['test'], label=method)
            axes[1, 0].plot(metrics[method]['scores']['npv_precision_avg']['train'], label=method)
            axes[1, 1].plot(metrics[method]['scores']['npv_precision_avg']['test'], label=method)

        perms = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for i, j in perms:
            axes[i, j].legend()
            axes[i, j].set_xlabel("Epochs")

        axes[1, 0].set_ylim(top=1.0)
        axes[1, 1].set_ylim(top=1.0)

        fig.set_tight_layout(True)
        fig.show()

        fig = plt.figure()
        fig.suptitle("NN Training Time (s)")
        times = [metrics[method]['time'] for method in methods]
        plt.bar(methods, times)
        fig.show()

    step_fns = [step1, step2, step3, step4, step5]
    render_fns = [render_step1, render_step2, render_step3, render_step4, render_step5]
