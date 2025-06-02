import torch
import hydra
from pathlib import Path
from hydra.utils import instantiate
from ClimatExML.loader import ClimatExEmulatorDataModule
from hrstream import HRStreamEmulator

from utils import setup_logger, validate_model_path, run_inference, save_output_to_zarr


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg):

    ensemble_size = 1

    logger = setup_logger()

    # Instantiate data
    emulation_data = instantiate(cfg.emulator.emulation_data)
    invariant = instantiate(cfg.invariant)

    # Instantiate DataModule
    emulator_dm = ClimatExEmulatorDataModule(
        emulation_data=emulation_data,
        invariant=invariant,
        batch_size=1,
        num_workers=cfg.hardware.num_workers,
    )
    emulator_dm.setup()

    # Grab data loader
    emulation_loader = emulator_dm.emulation_dataloader()

    model_path = Path(validate_model_path())
    logger.info(f"Emulating using: {model_path.stem}")

    # Only have the one Emulator for now, but possibly move the decleration to
    # a config file.
    emulator = HRStreamEmulator(
        model_path=model_path,
        device=cfg.hardware.accelerator,
    )

    logger.info("Running inference over the provided LR data set...")

    outputs = run_inference(emulator=emulator,
                            emulation_loader=emulation_loader,
                            ensemble_size=ensemble_size,
                            logger=logger)

    output_variables = cfg.emulator.output_variables
    save_output_to_zarr(output=outputs,
                        output_variables=output_variables,
                        model_path=model_path,
                        emulation_loader=emulation_loader,
                        logger=logger)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    torch.cuda.empty_cache()
    main()
