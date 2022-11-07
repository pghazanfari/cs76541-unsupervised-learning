import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping
import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoricalEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, df, categorical_cols):
        super().__init__()
        real_cols = [c for c in df.columns if c not in categorical_cols]
        self.X = torch.tensor(df[real_cols].to_numpy()).float()
        self.y = torch.tensor(df[categorical_cols].to_numpy()).float()

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
    def __len__(self):
        return self.X.shape[0]

class CategoricalEmbedding(pl.LightningModule):
    def __init__(self, in_features, out_features, hidden_layer_size=256):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(in_features, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, out_features),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.nn(X)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.nn(x)
        return {'loss': F.binary_cross_entropy(y_pred, y)}

    def training_step_end(self, outputs):
        self.log('train_loss', outputs['loss'], on_step=False, on_epoch=True, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

class CategoricalEmbeddingEncoder:
    def __init__(self, categorical_cols):
        self.categorical_cols = categorical_cols

    def fit(self, df, batch_size=64, num_workers=8, max_epochs=1000):
        dataset = CategoricalEmbeddingDataset(df, self.categorical_cols)
        real_cols = [c for c in df.columns if c not in self.categorical_cols]
        self.embedding = CategoricalEmbedding(len(real_cols), len(self.categorical_cols))
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[EarlyStopping('train_loss', min_delta=0.001)])
        #trainer = pl.Trainer(max_epochs=max_epochs)
        trainer.fit(model=self.embedding, train_dataloaders=train_dataloader)

    def predict(self, df):
        result = df.copy()
        real_cols = [c for c in df.columns if c not in self.categorical_cols]
        with torch.no_grad():
            result[self.categorical_cols] = self.embedding(torch.tensor(df[real_cols].to_numpy()).float()).numpy()
        return result
