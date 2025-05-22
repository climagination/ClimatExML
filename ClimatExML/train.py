import comet_ml
import lightning as pl
from ClimatExML.wgan_gp import SuperResolutionWGANGP
from ClimatExML.cnn import CNNTrainer
from ClimatExML.loader import ClimatExLightning
from ClimatExML.mlclasses import InputVariables, InvariantData
from lightning.pytorch.loggers import CometLogger
import torch
import logging
import hydra
from hydra.utils import instantiate
import os
import warnings


@hydra.main(config_path="conf", config_name="config")
def main(cfg: dict):
    hyperparameters = cfg.hyperparameters
    tracking = cfg.tracking
    hardware = cfg.hardware

    comet_logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name=tracking.project_name,
        workspace=tracking.workspace,
        save_dir=tracking.save_dir,
        experiment_name=tracking.experiment_name,
    )

    comet_logger.log_hyperparams(cfg.hyperparameters)

    # These are objects instantiated with config information (see config.yaml)
    train_data = instantiate(cfg.train_data)
    validation_data = instantiate(cfg.validation_data)
    invariant = instantiate(cfg.invariant)

    clim_data = ClimatExLightning(
        train_data,
        validation_data,
        invariant,
        hyperparameters.batch_size,
        num_workers=hardware.num_workers,
    )

    sr_trainers = {
        "cnn": CNNTrainer(
                tracking,
                hardware,
                hyperparameters,
                invariant,
                log_every_n_steps=tracking.log_every_n_steps,
            ),
        "wgan": SuperResolutionWGANGP(
                tracking,
                hardware,
                hyperparameters,
                invariant,
                log_every_n_steps=tracking.log_every_n_steps,
            )
    }

    srmodel = sr_trainers[cfg.trainer]

    trainer = pl.Trainer(
        precision=hardware.precision,
        accelerator=hardware.accelerator,
        max_epochs=hyperparameters.max_epochs,
        logger=comet_logger,
        detect_anomaly=False,
        devices=hardware.devices,
        strategy=hardware.strategy,
        check_val_every_n_epoch=1,
        log_every_n_steps=tracking.log_every_n_steps,
    )

    trainer.fit(srmodel, datamodule=clim_data)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    torch.cuda.empty_cache()
    logging.basicConfig(level=logging.INFO)
    main()
