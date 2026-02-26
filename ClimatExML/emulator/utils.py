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
import json
from typing import Dict
from glob import glob


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
    Save model output to a zarr dataset with geospatial coordinates.

    Args:
        output (Tensor): Model output of shape [T, N, C, H, W] or [T, C, H, W].
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

    # Load grid coordinates
    grid = load_grid_coordinates(model_path, logger)

    for c, var in enumerate(output_variables):
        data = output[:, :, c, :, :].cpu().numpy()  # shape [T, N, H, W]

        # Load variable-specific metadata
        metadata = load_metadata_json(var, model_path)
        units = metadata.get("units", {}).get("original", "unknown")
        hash_ = metadata.get("hash", "unknown")

        # Construct attributes
        attrs = {
            "variable": var,
            "units": units,
            "preprocessing_hash": hash_,
            "emulator_model": model_path.stem,
        }

        # Create base coordinates
        coords = {
            "time": datetimes,
            "realization": np.arange(N),
            "rlat": np.arange(H),
            "rlon": np.arange(W),
        }

        # Add lat/lon coordinates if grid is available
        if grid is not None:
            coords["lat"] = (["rlat", "rlon"], grid.lat.values)
            coords["lon"] = (["rlat", "rlon"], grid.lon.values)

        da = xr.DataArray(
            data,
            dims=("time", "realization", "rlat", "rlon"),
            coords=coords,
            attrs=attrs
        )

        ds = xr.Dataset({var: da})

        zarr_path = get_save_path_from_example_path(example_path, var, model_path)
        ds.to_zarr(zarr_path, mode="w")
        logger.info(f"✅ Saved '{var}' to {zarr_path} with geospatial coordinates.")


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


def load_metadata_json(var_name: str, model_path: Path) -> Dict:
    """
    Load normalization metadata JSON for a given variable.

    Args:
        var_name (str): Name of the output variable (e.g., "uas").
        model_path (Path): Path to the model file (used to locate metadata dir).

    Returns:
        dict: Parsed metadata dictionary.
    """
    metadata_dir = model_path.parent / "../feature_scaling_metadata"
    json_files = glob(str(metadata_dir / f"hr_{var_name}_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No metadata file found for variable '{var_name}' in {metadata_dir}")

    with open(json_files[0], "r") as f:
        return json.load(f)


def un_normalize(output: torch.Tensor, output_variables: List[str], model_path: Path) -> torch.Tensor:
    """
    Un-normalize model output using metadata JSON files for each output variable.

    Args:
        output (torch.Tensor): Model output of shape [T, N, C, H, W].
        output_variables (List[str]): Names of the output variables (length must match C).
        model_path (Path): Path to the model used for inference.

    Returns:
        torch.Tensor: Un-normalized tensor with same shape.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"🔄 Starting un-normalization for {len(output_variables)} variable(s)...")

    unnormalized = []

    for i, var in enumerate(output_variables):
        stats = load_metadata_json(var, model_path)
        logger.info(f"📦 Loaded metadata for '{var}' from model directory.")

        var_tensor = output[:, :, i, :, :]  # [T, N, H, W]

        if stats.get("apply_normalize", False):
            min_ = stats["min"]
            max_ = stats["max"]
            logger.info(f"🔧 Un-normalizing '{var}' using min={min_}, max={max_}")
            var_tensor = var_tensor * (max_ - min_) + min_

        elif stats.get("apply_standardize", False):
            mean_ = stats["mean"]
            std_ = stats["std"]
            logger.info(f"🔧 Un-standardizing '{var}' using mean={mean_}, std={std_}")
            var_tensor = var_tensor * std_ + mean_

        else:
            logger.warning(f"⚠️ No normalization method flagged for '{var}'. Skipping transform.")

        unnormalized.append(var_tensor.unsqueeze(2))  # restore C dim

    logger.info("✅ Un-normalization complete.")
    return torch.cat(unnormalized, dim=2)  # [T, N, C, H, W]


def load_grid_coordinates(model_path: Path, logger: logging.Logger) -> Optional[xr.Dataset]:
    """
    Load grid coordinates from the target grid NetCDF file.

    Args:
        model_path (Path): Path to the model file.
        logger (logging.Logger): Logger for status updates.

    Returns:
        xr.Dataset or None: Grid dataset with lat/lon coordinates, or None if not found.
    """
    # Look for grid file in the expected location
    grid_dir = model_path.parent / "../grid"

    # Try to find grid file matching the model name pattern
    grid_files = list(grid_dir.glob("*_target_grid.nc"))

    if not grid_files:
        logger.warning(f"⚠️ No grid file found in {grid_dir}. Output will not have lat/lon coordinates.")
        return None

    if len(grid_files) > 1:
        logger.warning(f"⚠️ Multiple grid files found. Using first: {grid_files[0].name}")

    grid_file = grid_files[0]
    logger.info(f"📐 Loading grid coordinates from {grid_file.name}")

    try:
        grid = xr.open_dataset(grid_file)
        logger.info(f"   Grid dimensions: {dict(grid.sizes)}")
        return grid
    except Exception as e:
        logger.error(f"❌ Failed to load grid file: {e}")
        return None
