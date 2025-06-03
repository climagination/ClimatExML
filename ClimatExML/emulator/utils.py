import os
import logging
from typing import Optional
import torch
import re
import xarray as xr
from tqdm import tqdm
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from typing import List


def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with the specified name and logging level.

    Args:
        name (str): Name of the logger.
        level (int): Logging level, e.g., logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logging.basicConfig(level=level)
    return logger


def get_output_path(file_name: str, subdir: str = "inference_outputs") -> str:
    """
    Construct a full path to the output file within a subdirectory under DATA_DIR.

    Args:
        file_name (str): Name of the output file (e.g., "test_output.pt").
        subdir (str): Subdirectory name under DATA_DIR to store the output.

    Returns:
        str: Full path to the output file.
    """
    output_dir = os.path.join(os.environ.get("DATA_DIR", "./data"), subdir)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, file_name)


def validate_model_path() -> str:
    """
    Validate and return the model path from the MODEL_PATH environment variable.

    Raises:
        AssertionError: If MODEL_PATH is not set or does not point to a .pt file.

    Returns:
        str: Validated path to the TorchScript model file.
    """
    model_path = os.environ.get("MODEL_PATH")
    assert model_path is not None, "Set MODEL_PATH environment variable to a .pt TorchScript model"
    assert model_path.endswith(".pt"), "Expecting TorchScript model with .pt extension"
    return model_path


def run_inference(emulator, dataloader, ensemble_size: int, logger: Optional[logging.Logger] = None) -> torch.Tensor:
    """
    Run inference using an emulator over all batches in a dataloader.

    Args:
        emulator: An instance of BaseEmulator or subclass implementing `.sample()`.
        dataloader: A PyTorch DataLoader yielding (lr, _, hr_invariant) batches.
        ensemble_size (int): Number of samples to generate per input.
        logger (Optional[logging.Logger]): Logger to log output shape info.

    Returns:
        torch.Tensor: Tensor of shape [T, N, C, H, W], where T is time, N is ensemble.
    """
    all_outputs = []

    for batch in tqdm(dataloader, desc="Running inference"):
        lr, hr_inv = batch
        output = emulator.sample(lr, hr_inv, ensemble_size)

        # Ensure ensemble dimension is retained
        if output.ndim == 4:  # [N, C, H, W] → [1, N, C, H, W]
            output = output.unsqueeze(0)

        all_outputs.append(output)

    full_output = torch.cat(all_outputs, dim=0)

    if logger:
        logger.info(f"Final stacked output shape: {full_output.shape}")

    return full_output


def save_output_to_zarr(
    output: torch.Tensor,
    output_variables: List[str],
    model_path: Path,
    emulation_loader: DataLoader,
    logger: logging.Logger,
) -> None:
    """
    Save model output to a zarr dataset.

    Args:
        output (Tensor): Model output of shape [T, N, C, H, W] or [T, C, H, W].
        datetimes (List[datetime64]): List of datetime64 timestamps of length T.
        output_variables (List[str]): List of variable names corresponding to C channels.
        model_path (Path): Path to the model file, used for naming the output directory.
        emulation_loader (DataLoader): DataLoader that holds data filepaths.
        logger (Logger): Logger object for status updates.
    """
    lr_paths = emulation_loader.dataset.lr_paths[0]
    example_path = Path(lr_paths[0])

    datetimes = extract_datetimes_from_filenames(lr_paths)

    if output.ndim == 4:
        output = output.unsqueeze(1)  # Expand to [T, 1, C, H, W]

    T, N, C, H, W = output.shape
    assert C == len(output_variables), "Mismatch between output channels and variable names"
    assert T == len(datetimes), "Mismatch between time dimension and datetime list"

    for c, var in enumerate(output_variables):
        data = output[:, :, c, :, :].cpu().numpy()  # shape [T, N, H, W]

        ds = xr.Dataset(
            {
                var: (("time", "realization", "rlon", "rlat"), data),
            },
            coords={
                "time": datetimes,
                "realization": np.arange(data.shape[1]),
                "rlon": np.arange(data.shape[2]),
                "rlat": np.arange(data.shape[3]),
            },
        )
        zarr_path = get_save_path_from_example_path(example_path, var, model_path)
        ds.to_zarr(zarr_path, mode="w")
        logger.info(f"Saved variable '{var}' to {zarr_path}")


def extract_datetimes_from_filenames(filenames: List[str]) -> np.ndarray:
    """
    Extract datetime64[ns] array from filenames like 'uas_2000-10-01-06.pt'.
    
    Args:
        filenames (List[str]): List of file paths or names.
    
    Returns:
        np.ndarray: Array of datetime64[ns] values.
    """
    datetimes = []
    for name in filenames:
        name = Path(name)
        match = re.search(r"\d{4}-\d{2}-\d{2}-\d{2}", str(name))
        if match:
            date_str = match.group()  # '2000-10-01-06'
            date_part, hour = date_str.rsplit("-", 1)
            dt = np.datetime64(f"{date_part}T{hour}:00:00", "ns")
            datetimes.append(dt)
        else:
            raise ValueError(f"No valid datetime found in filename: {name}")
    
    return np.array(datetimes)


def get_save_path_from_example_path(
    example_path: Path, output_variable: str, model_path: Path
) -> Path:
    """
    Construct the .zarr save path for a given variable and model.

    Args:
        example_path (Path): Path to a sample input file.
        output_variable (str): Output variable name.
        model_path (Path): Trained model file path.

    Returns:
        Path: Full path to the .zarr output file.
    """
    model_name = model_path.stem
    dataset_split = example_path.parents[2]  # e.g., .../train or .../validation
    return dataset_split / output_variable / f"hr_inf_{model_name}.zarr"