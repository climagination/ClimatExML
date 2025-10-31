# Deploying on DRAC HPC Clusters

**Last Updated:** October 2025  
**Tested On:** Nibi cluster  
**Author:** Seamus Beairsto

:::{note}
This guide is specific to the **Nibi cluster** on DRAC. Other clusters (Trillium, Rorqual, Fir, Narval) may have different module versions and configurations. This will be updated as testing on other clusters is completed.
:::

## Prerequisites

- Active DRAC/Compute Canada account with access to Nibi
- Project allocation with GPU resources (e.g., `def-yourpi`)
- SSH access to the cluster
- (Some) Familiarity with SLURM job submission

## Environment Setup on DRAC

### 1. Choose Installation Location

DRAC clusters have different storage areas with specific purposes:

- **`/home/username/`** - 50GB quota, backed up, for code and scripts
- **`/project/def-yourpi/username/`** - Larger quota, shared with group, for long-term storage
- **`/scratch/username/`** - Large temporary storage (60 days), for temporary datasets and outputs

**Recommended:** Install ClimatExML in your project space:

```bash
cd /home/username/projects/def-monahana/username/
mkdir -p ClimatEx
cd ClimatEx
```

### 2. Load Required Modules
On Nibi, only the CUDA module is required:

```bash
module purge  # Clear any conflicting modules
module load cuda/12.9
```

:::{note}
The system Python 3 is sufficient for creating virtual environments. Unlike some clusters, you don't need to load a separate Python module.
:::

### 3. Create Virtual Environment and Install ClimatExML
```bash
# Create virtual environment
python3 -m venv climatexvenv
source climatexvenv/bin/activate

# Clone repository
git clone https://github.com/nannau/ClimatExML
cd ClimatExML

# Install ClimatExML with CUDA 12.9 support
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
pip install -e .
```

:::{warning}
Important: Ensure your PyTorch installation matches the CUDA module version you'll use in jobs. See the troubleshooting section below for details on CUDA version mismatches.
:::

### 4. Verify GPU Access
GPU verification must be done within a job submission or interactive session, as GPUs are not available on login nodes. See Interactive Sessions for detailed instructions on requesting an interactive GPU session.

Quick verification commands (run within a GPU job):

``` bash
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'PyTorch CUDA version: {torch.version.cuda}')"
```

## Batch Job Submission
### Complete Working Example
Here is a complete, tested batch submission script for running ClimatExML on Nibi:

``` bash
#!/bin/bash

##############################
#       SLURM SETTINGS       #
##############################

#SBATCH --account=def-monahana           # PI name
#SBATCH --mail-user=sbeairsto@uvic.ca    # Your email
#SBATCH --mail-type=BEGIN,END,FAIL       # When to send emails

#SBATCH --job-name=climatex_training     # Job name
#SBATCH --nodes=1                        # Number of nodes, usually 1
#SBATCH --ntasks=1                       # Number of tasks, usually 1
#SBATCH --cpus-per-task=8                # Number of CPUs, scales with batch size
                                         # Ensure 'num_workers' in DataLoader <= this number
#SBATCH --mem=80G                        # RAM, scales with batch size
#SBATCH --time=00-15:10:00               # Requested time (DD-HH:MM:SS)

#SBATCH --gres=gpu:h100:1                # Requesting 1 H100 GPU

##############################
#     JOB SETUP & LOGGING    #
##############################

# Define output log file using SLURM job ID
OUTPUT_FILE="/home/seamus/scratch/gan/run_${SLURM_JOB_ID}.log"

##############################
#    ENVIRONMENT SETUP       #
##############################

nvidia-smi
module purge
module load cuda/12.9

# Activate virtual environment
source /home/seamus/projects/def-monahana/sbeairsto/ClimatEx/ClimatExML/climatexvenv/bin/activate
cd /home/seamus/projects/def-monahana/sbeairsto/ClimatEx/ClimatExML

##############################
#       DATA TRANSFER        #
##############################

# Copy and extract dataset to $SLURM_TMPDIR
# Note: $SLURM_TMPDIR is fast local SSD storage. Always use this for data during jobs.

cp /path/to/my/data.tar.gz $SLURM_TMPDIR

# Set threads for parallel decompression
PIGZ_THREADS=${SLURM_CPUS_PER_TASK:-1}
echo "Using $PIGZ_THREADS threads for pigz decompression."

# Extract into SLURM_TMPDIR
mkdir -p "$SLURM_TMPDIR"
tar -C "$SLURM_TMPDIR" -I "pigz -d -p $PIGZ_THREADS" -xvf "$SLURM_TMPDIR/data.tar.gz"

echo "Files unzipped into $SLURM_TMPDIR/data"

# Verify extraction
pigz -t "$SLURM_TMPDIR/data.tar.gz" || echo "gzip test failed"
tar -tf "$SLURM_TMPDIR/data.tar.gz" | head -n 50

# Show directory structure
find "$SLURM_TMPDIR" -maxdepth 3 -type d -print

##############################
#     RUN TRAINING SCRIPT    #
##############################

# Set environment variables if needed
. env_vars.sh

# Run training
python ClimatExML/train.py > "$OUTPUT_FILE" 2>&1
```


### Key Points

-   **`$SLURM_TMPDIR`**: Always copy and extract data here for fast I/O (local SSD vs. networked storage)
-   **`pigz`**: Parallel gzip for faster decompression using multiple CPUs
-   **`--cpus-per-task`**: Should match or exceed  `num_workers`  in your PyTorch DataLoader
-   **`--mem`**: Scale with batch size; H100 GPUs pair well with 80G+ RAM

For detailed explanations of SLURM parameters and job submission strategies, see [Batch Jobs](https://radia.vrd-drv.crc.ca/c/batch.md).

## Common Issues and Solutions

### Issue: No CUDA Connection in Interactive SLURM Job

When running an interactive job on DRAC, I was able to confirm that a GPU was available using `nvidia-smi`. However, when testing Python GPU connectivity in my virtual environment, I received `torch.cuda.is_available() = False`.

#### Debugging Steps

**Step 1:** I loaded the DRAC `cuda` module, expecting that would resolve the issue. It did not.

**Step 2:** I checked the CUDA version of the loaded module (CUDA 12.9) and compared it to the CUDA version built into my PyTorch installation (`torch.version.cuda`), which was 12.4. There was a mismatch.
``` bash
module load cuda/12.9
python -c "import torch; print(f'Module CUDA: 12.9, PyTorch CUDA: {torch.version.cuda}')"
# Output: Module CUDA: 12.9, PyTorch CUDA: 12.4
```

**Step 3:** I then loaded the DRAC CUDA module for version 12.4 to match my PyTorch build, but the issue persisted.

**Step 4:** I reinstalled PyTorch in my virtual environment, this time targeting CUDA 12.9 to match the loaded module:

``` bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129` 
```

**Step 5:** After pairing the DRAC CUDA 12.9 module with the CUDA 12.9 PyTorch build, `torch.cuda.is_available()` returned `True`, confirming that Python could now access the GPU.

#### Key Learning

This issue was caused by a **mismatch between the CUDA version in my Python environment and the version provided by the DRAC CUDA module**. DRAC's default `pip install` process may install PyTorch with a different CUDA version than what's available via modules.

:::{tip} **Always match your PyTorch CUDA version to the loaded CUDA module.** Check versions with:

``` bash
# Check loaded module version
module list

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"` 
```

Install PyTorch with the matching CUDA version:

``` bash
# For CUDA 12.9
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129

# For CUDA 12.4
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

# For CUDA 11.8
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118` 
```
:::

### Issue: Job Fails Due to Data Path

**Problem:** Training script can't find data even though it was copied to `$SLURM_TMPDIR`

**Solution:** Ensure your training configuration uses the correct path. Either:

1.  Update your config file to use  `$SLURM_TMPDIR`:

``` python
data_path = os.path.join(os.environ['SLURM_TMPDIR'], 'cropped_pt_and_metadata')
```

2.  Or set an environment variable in your job script:

``` bash
export DATA_PATH="$SLURM_TMPDIR/cropped_pt_and_metadata"
```


### Issue: Module Conflicts

**Problem:** Errors about incompatible or missing libraries

**Solution:** Always start with `module purge` to clear any conflicting modules from your environment:

``` bash
module purge
module load cuda/12.9` 
```

## Monitoring and Managing Jobs

``` bash
# View your queued/running jobs
squeue -u $USER

# Cancel a job
scancel JOBID

# View detailed job info
scontrol show job JOBID

# Check job efficiency after completion
seff JOBID` 
```

For more details on monitoring and optimization, see [Optimization](https://radia.vrd-drv.crc.ca/c/optimization.md).

## Storage and Data Management

### Storage Locations

| Location | Path | Purpose | Quota | Backup | Purge |
|----------|------|---------|-------|--------|-------|
| Home | `/home/username/` | Code, configs | ~50GB | Yes | No |
| Project | `/home/username/projects/def-pi/` | Shared data, models | Varies | Yes | No |
| Scratch | `/home/username/scratch/` | Temporary datasets | > 1 TB | No | 60 days |
| SLURM temp | `$SLURM_TMPDIR` | Per-job fast storage | Node-dependent | No | Job end |

### Best Practices

1.  **Code and environments**:  `/home/`  or  `/projects/`
2.  **Large datasets**:  `/projects/`  (as compressed tar.gz)
3.  **Active job data**: Copy to  `$SLURM_TMPDIR`  at job start
4.  **Model output and checkpoints**: Save to  `/scratch/`  during training, move important ones to  `/project/

 Check out the [DRAC documentation](https://docs.alliancecan.ca/wiki/Storage_and_file_management) for more details.

## Performance Notes

Based on testing with ClimatExML on Nibi's H100 GPUs:

-   **Data Loading**: Using  `$SLURM_TMPDIR`  provides 5-10x faster I/O compared to reading directly from  `/scratch/`  or  `/project/`
-   **Decompression**: Parallel decompression with  `pigz`  significantly speeds up data preparation
-   **Optimal Configuration**:  `--cpus-per-task=8`  with  `--mem=80G`  works well for most training runs

## Resources

-   [DRAC/Alliance Documentation](https://docs.alliancecan.ca/)
-   [CUDA Compatibility Guide](https://docs.nvidia.com/deploy/cuda-compatibility/)
-   {doc}`drac-interactive`
-   {doc}`drac-batch`
-   {doc}`drac-optimization`