import comet_ml
import lightning as pl
from ClimatExML.wgan_gp_stoch import SuperResolutionWGANGP
from ClimatExML.loader import ClimatExLightning
from ClimatExML.mlclasses import InputVariables, InvariantData
from lightning.pytorch.loggers import CometLogger
import torch
import logging
import hydra
from hydra.utils import instantiate
import os
import warnings
#from lightning.pytorch import seed_everything

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
    #seed_everything(42,workers=True) #MANUAL RANDOM SEEDS
    clim_data = ClimatExLightning(
        train_data,
        validation_data,
        invariant,
        hyperparameters.batch_size,
        num_workers=hardware.num_workers,
    )

    srmodel = SuperResolutionWGANGP(
        tracking,
        hardware,
        hyperparameters,
        invariant,
        log_every_n_steps=tracking.log_every_n_steps,
    )

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


def check_env_vars():
    if os.environ.get("OUTPUT_DIR") is None:
        # make a warning
        warnings.warn(
            "OUTPUT_DIR is not set. Defaulting to current directory. This is likely not what you want!"
        )
        os.environ["OUTPUT_DIR"] = os.getcwd()


if __name__ == "__main__":

    # check that the expected environment variables are set

    check_env_vars()
    torch.set_float32_matmul_precision("medium")
    torch.cuda.empty_cache()
    logging.basicConfig(level=logging.INFO)
    main()
