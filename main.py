import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig
import pytorch_lightning as pl

import wandb

from src.callbacks.callbacks import ImagePredictionLogger

@hydra.main(config_path="config/", config_name="config.yaml", version_base='1.3')
def main(cfg: DictConfig):

    model = instantiate(cfg.model) 

    data_module = instantiate(cfg.data)
    data_module.prepare_data()
    data_module.setup()

    val_samples = next(iter(data_module.val_dataloader()))

    classifier = instantiate(cfg.lightning_module)

    wandb.login(key="ad19c5aa5d4bff28d21f9939714e6f7c8e81a1b7")

    logger = instantiate(cfg.logger)

    callbacks = []
    for _, cb_conf in cfg.callbacks.items():
        callbacks.append(instantiate(cb_conf))

    callbacks.append(ImagePredictionLogger(val_samples))


    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=cfg.trainer.max_epochs,
                     accelerator=cfg.trainer.accelerator,
                     devices = cfg.trainer.devices,
                     logger = logger,
                     callbacks=callbacks,
                     log_every_n_steps = cfg.trainer.log_every_n_steps
                     )

    # Train the model
    trainer.fit(model=classifier, datamodule=data_module)

    # Evaluate the model on the held out test set ⚡⚡
    trainer.test(model=classifier, datamodule=data_module, ckpt_path=cfg.trainer.ckpt_path)

    wandb.finish()



if __name__ == "__main__":
    main()

