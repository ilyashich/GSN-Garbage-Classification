import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import seaborn as sn
import pandas as pd

import pytorch_lightning as pl
import torchmetrics

import wandb


class EfficientNetModule(pl.LightningModule):
    def __init__(self, model, num_classes, learning_rate=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = model
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, x, y): 
        return F.cross_entropy(x, y)

    def common_step(self, batch, batch_idx):
        x, y = batch
        y_p = self(x)
        loss = self.compute_loss(y_p,y)
        return loss, y_p, y

    def common_test_valid_step(self, batch, batch_idx):
        loss, y_p, y = self.common_step(batch, batch_idx)
        preds = torch.argmax(y_p, dim=1)
        acc = torchmetrics.functional.accuracy(preds, y, num_classes = self.num_classes, task="multiclass")
        return loss, acc, y, preds

    def training_step(self, batch, batch_idx):
        loss, acc, _, _ = self.common_test_valid_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)    
        return {'loss':loss, 'train_acc':acc}

    def validation_step(self, batch, batch_idx):
        loss, acc, y, preds = self.common_test_valid_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True) 
        return {'val_loss':loss, 'val_acc': acc, 'labels': y, 'outputs': preds}
        
    def test_step(self, batch, batch_idx):
        loss, acc, y, preds = self.common_test_valid_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss':loss, 'test_acc': acc, 'labels': y, 'outputs': preds}
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_lambda = lambda epoch: 0.99 ** epoch
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) 
        return [optimizer], [lr_scheduler] 