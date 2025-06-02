# Emulator

This module runs inference using a trained super-resolution model to generate high-resolution climate outputs from low-resolution inputs. The outputs are saved as `.zarr` datasets for easy downstream analysis and model evaluation.

---

## 🔧 Requirements

This module relies on the following libraries (in addition to those included in `setup.py`):

-  `xarray`
-  `zarr`
-  `tqdm`
---

## 🚀 Running the Emulator

Ensure you’ve set the following environment variables:

```bash

export  MODEL_PATH=/absolute/path/to/your_model.pt

export  DATA_DIR=/absolute/path/to/your/data

```
Then run:

```bash

python  emulator/test_inference.py

```

## ⚙️ Config Notes

Edit conf/config.yaml to specify:


- emulation_data.lr_paths: List of glob patterns for LR input files (e.g., uas, vas)
- output_variables: List of HR variable names your model predicts (e.g., ["uas", "vas"])
- hardware.accelerator: "gpu" or "cpu"

----------

## 💾 Output Format

-   Output shape: `[T, N, C, H, W]` where `T = time`, `N = ensemble`, `C = channels`
    
-   Each output channel is saved to a separate `.zarr` file under:
 ```
/DATA_DIR/{split}/{variable}/hr_inf_{model_name}.zarr
```
-   The time axis is extracted from the input file names (e.g., `uas_2000-10-01-06.pt` → `2000-10-01T06:00:00`)
    
----------

## File Structure

```text

emulator/
├── emulate.py       # Main entry point: loads model, runs inference, saves output
├── utils.py                # Helper functions: logging, path generation, zarr saving
├── base.py                 # Abstract base class for emulators (device handling + API)
├── hrstream.py             # Concrete implementation of BaseEmulator for TorchScript models
```
----------

## ✅ Example Output

```text

`/home/sbeairsto/data-sbeairsto/vancouver_island/validation/uas/hr_inf_generator_cnn-two-variable/uas_hr_inf_generator_cnn-two-variable.zarr` 
```
This file contains an `xarray` dataset with dimensions:

-   `time`  
-   `rlon`
-   `rlat`
    
----------

## 🔮 Future Work
- Introduce a dedicated `EmulatorDataLoader` that directly streams `.zarr` files using `xarray` or `zarr-python` backend, enabling batch inference over chunked time windows without requiring pre-saved `.pt` files.