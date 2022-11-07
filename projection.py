import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.cm as cm

class PairwiseDataset(torch.utils.data.Dataset):
    def __init__(self, x1, x2):
        super().__init__()
        self.x1 = x2
        self.x2 = x2

    def __getitem__(self, i):
        return self.x1[i // self.x2.shape[0]], self.x2[i % self.x2.shape[0]]
    
    def __len__(self):
        return self.x1.shape[0] * self.x2.shape[0]

class Projector(pl.LightningModule):
    def __init__(self, in_features, hidden_layer_size=128, out_features=2, distance_fn=F.pairwise_distance):
        super().__init__()
        self.distance_fn = distance_fn
        self.nn = nn.Sequential(
            nn.Linear(in_features, hidden_layer_size), 
            nn.ReLU(), 
            nn.Linear(hidden_layer_size, out_features)
        )

    def forward(self, X):
        return self.nn(X)

    def training_step(self, batch, batch_idx):
        x1, x2 = batch
        y = self.distance_fn(x1, x2)
        
        y1_pred = self.nn(x1)
        y2_pred = self.nn(x2)
        y_pred = self.distance_fn(y1_pred, y2_pred)
        return {'loss': F.mse_loss(y_pred, y)}

    def training_step_end(self, outputs):
        self.log('train_loss', outputs['loss'], on_step=False, on_epoch=True, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

class MultivariableClusterVisualizer:
    def __init__(self, projector):
        self.projector = projector

    def fit(self, x1, x2=None, batch_size=64, num_workers=8, max_epochs=100):
        if x2 is None:
            x2 = x1
        dataset = PairwiseDataset(x1, x2)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[EarlyStopping('train_loss', min_delta=0.1)])
        trainer.fit(model=self.projector, train_dataloaders=train_loader)

    def render(self, X, estimator, cluster_centers=None, fig=None):
        fig = fig or plt.figure()
        with torch.no_grad():
            y1 = self.projector.nn(X).numpy()
            if cluster_centers is not None:
                y2 = self.projector.nn(cluster_centers).numpy()
            
        cluster_indices = estimator.predict(X)

        clusters = list(sorted(np.unique(cluster_indices)))

        for i in clusters:
            indices = np.argwhere(cluster_indices == i).flatten()
            yi = np.take(y1, indices, axis=0)
            color = cm.nipy_spectral(float(i) / len(clusters))
            plt.scatter(yi[:, 0], yi[:, 1], color=color)

        if cluster_centers is not None:
            plt.scatter(y2[:, 0], y2[:, 1], color='r')
        plt.show()