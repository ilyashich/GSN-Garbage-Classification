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
    
    def validation_epoch_end(self, outs):
        labels = torch.cat([o['labels'] for o in outs])
        outputs = torch.cat([o['outputs'] for o in outs])


        class_names=["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        cf_matrix = torchmetrics.functional.classification.confusion_matrix(outputs, labels, num_classes=self.num_classes, task="multiclass").cpu().numpy()
        #disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix / np.sum(cf_matrix, axis=1)[:, None], display_labels=class_names)
        dataframe = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in class_names],
                         columns = [i for i in class_names])
        plt.figure(figsize = (8,6))
        sn.heatmap(dataframe, annot=True)

        plt.title("Confusion Matrix")
        plt.ylabel("True Class"), 
        plt.xlabel("Predicted Class")

        self.logger.experiment.log({"media/val_confusion_matrix": wandb.Image(plt)})
        plt.close()
        
    def test_step(self, batch, batch_idx):
        loss, acc, _, _ = self.common_test_valid_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss':loss, 'test_acc': acc}
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer