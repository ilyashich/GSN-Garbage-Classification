import torch
import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

import wandb

from src.callbacks.callbacks import ImagePredictionLogger, ConfusionMatrixLogger
from pytorch_lightning.callbacks import LearningRateMonitor

@hydra.main(config_path="config/", config_name="config.yaml", version_base='1.3')
def main(cfg: DictConfig):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_module = instantiate(cfg.data)

    classifier = instantiate(cfg.lightning_module)

    wandb.login(key=cfg.wandb_key)

    logger = instantiate(cfg.logger)

    callbacks = []
    for _, cb_conf in cfg.callbacks.items():
        callbacks.append(instantiate(cb_conf))

    callbacks.append(ImagePredictionLogger())
    callbacks.append(ConfusionMatrixLogger())
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))


    # Initialize a trainer
    trainer = pl.Trainer(
                    **OmegaConf.to_container(cfg.trainer),
                    accelerator = device,
                    logger = logger,
                    callbacks=callbacks
    )

    # Train the model
    trainer.fit(model=classifier, datamodule=data_module)

    # Evaluate the model on the held out test set ⚡⚡
    trainer.test(model=classifier, datamodule=data_module, ckpt_path=cfg.ckpt_path)

    wandb.finish()



if __name__ == "__main__":
    main()

