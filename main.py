import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig
import pytorch_lightning as pl

import wandb

from src.callbacks.callbacks import ImagePredictionLogger

@hydra.main(config_path="config/", config_name="config.yaml")
def main(cfg: DictConfig):

    model, image_size = instantiate(cfg.model) 

    data_module = instantiate(cfg.data)
    data_module.image_size = image_size 
    data_module.prepare_data()
    data_module.setup()

    val_samples = next(iter(data_module.val_dataloader()))

    classifier = instantiate(cfg.lightning_module)

    wandb.login(key="ad19c5aa5d4bff28d21f9939714e6f7c8e81a1b7")

    logger = instantiate(cfg.logger)

    early_stop_callback = instantiate(cfg.callbacks.early_stopping)

    checkpoint_callback = instantiate(cfg.callbacks.model_checkpoint)

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=cfg.trainer.max_epochs,
                     accelerator=cfg.trainer.accelerator,
                     devices = cfg.trainer.devices,
                     logger = logger,
                     callbacks=[checkpoint_callback, early_stop_callback, ImagePredictionLogger(val_samples)],
                     log_every_n_steps = cfg.trainer.log_every_n_steps
                     )

    # Train the model
    trainer.fit(model=classifier, datamodule=data_module)

    # Evaluate the model on the held out test set ⚡⚡
    trainer.test(model=classifier, datamodule=data_module, ckpt_path=cfg.trainer.ckpt_path)

    wandb.finish()



if __name__ == "__main__":
    main()

