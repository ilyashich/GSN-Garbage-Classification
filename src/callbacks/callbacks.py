import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd

import torchmetrics

import wandb



class ImagePredictionLogger(Callback):
    def __init__(self, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):

        if batch_idx == 0:
            x, y = batch

            classes = trainer.datamodule.dataset_train.subset.dataset.classes
            
            trainer.logger.experiment.log({
                "examples":[wandb.Image(img, caption=f"Truth: {classes[y_i]}, Predicted: {classes[y_p]}") 
                            for img, y_i, y_p in zip(x[:self.num_samples], 
                                                     y[:self.num_samples], 
                                                     outputs["outputs"][:self.num_samples])]
            })


class ConfusionMatrixLogger(Callback):
    def __init__(self):
        super().__init__()
        self.val_outs= {
            "outputs": [],
            "labels": []
        }
        self.test_outs = {
            "outputs": [],
            "labels": []
        }

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.val_outs["outputs"].append(outputs["outputs"])
        self.val_outs["labels"].append(outputs["labels"])

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.test_outs["outputs"].append(outputs["outputs"])
        self.test_outs["labels"].append(outputs["labels"])

    def on_validation_epoch_end(self, trainer, pl_module):
        labels = torch.cat(self.val_outs["labels"])
        outputs = torch.cat(self.val_outs["outputs"])

        classes = trainer.datamodule.dataset_train.subset.dataset.classes

        self.log_conf_matrix(name="media/val_confusion_matrix", labels=labels, outputs=outputs, num_classes=pl_module.num_classes, classes=classes, trainer=trainer)        

        self.val_outs["outputs"].clear()
        self.val_outs["labels"].clear()

    def on_test_epoch_end(self, trainer, pl_module):
        labels = torch.cat(self.test_outs["labels"])
        outputs = torch.cat(self.test_outs["outputs"])

        classes = trainer.datamodule.dataset_test.subset.dataset.classes

        self.log_conf_matrix(name="media/test_confusion_matrix", labels=labels, outputs=outputs, num_classes=pl_module.num_classes, classes=classes, trainer=trainer)        

        self.test_outs["outputs"].clear()
        self.test_outs["labels"].clear()

    def log_conf_matrix(self, name, labels, outputs, num_classes, classes, trainer):
        cf_matrix = torchmetrics.functional.classification.confusion_matrix(outputs, labels, num_classes=num_classes, task="multiclass").cpu().numpy()
        dataframe = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                         columns = [i for i in classes])
        plt.figure(figsize = (8,6))
        sn.heatmap(dataframe, annot=True)

        plt.title("Confusion Matrix")
        plt.ylabel("True Class"), 
        plt.xlabel("Predicted Class")

        trainer.logger.experiment.log({name: wandb.Image(plt)})
        plt.close()
        
early_stop_callback = EarlyStopping(
   monitor='val_loss',
   patience=10,
   mode='min'
)


MODEL_CKPT_PATH = 'model/'
MODEL_CKPT = 'model-{epoch:02d}-{val_loss:.2f}'


checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    dirpath=MODEL_CKPT_PATH,
    filename=MODEL_CKPT,
    save_top_k=3,
    mode='max'
)