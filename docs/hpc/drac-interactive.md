# Interactive Sessions on DRAC

**Tested on:** Nibi cluster  
**Last updated:** October 2025

Interactive sessions allow you to debug and test your code on a compute node before submitting larger batch jobs. This is similar to working on a lab machine (like Tars or Thufir), but on DRAC's compute infrastructure.

## Requesting an Interactive Session

Use `salloc` to request resources:

```bash
salloc --time=02:05:00 \
       --mem=50G \
       --ntasks=1 \
       --cpus-per-task=8 \
       --account=def-monahana \
       --job-name=gpu_interactive \
       --gres=gpu:h100:1 \
       --mail-user=sbeairsto@uvic.ca \
       --mail-type=BEGIN
```

**Parameters explained:**

-   `--time`: Maximum session duration (HH:MM
    
    format)
-   `--mem`: RAM allocation
-   `--cpus-per-task`: Number of CPU cores
-   `--account`: Your PI's allocation account
-   `--gres=gpu:h100:1`: Request 1 H100 GPU
-   `--mail-type=BEGIN`: Email notification when allocation is ready

Extra DRAC documentation can be found [here].(https://docs.alliancecan.ca/wiki/Running_jobs#Interactive_jobs)

::: {note} **Allocation wait times** vary from 1 minute to 10+ hours depending on requested resources and cluster availability. The email notification (`--mail-type=BEGIN`) is helpful for longer waits.
:::

## Checking Allocation Status

While waiting for your allocation, you can check its status:

``` bash
# Check your pending/running jobs
squeue -u $USER

# Output shows job ID, status, and remaining time
```

## Connecting to Your Allocation

Once your allocation is granted, attach to the compute node:

``` bash
srun --pty bash
```

You'll now have an interactive shell on the allocated compute node with access to the requested GPU.

## Setting Up Your Environment

Before running any GPU code, load the required modules:

```bash
module purge
module load cuda/12.9
source /path/to/your/climatexvenv/bin/activate
```

## Verifying GPU Access

``` bash
# Check GPU is available
nvidia-smi

# Test PyTorch GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Debugging with Data

### Option 1: Using Synthetic Data (Recommended for Quick Testing)

For rapid debugging without the overhead of copying and unpacking large datasets, use the synthetic data loader:

``` python
# In your training script or config
use_synthetic_data = True  # Enable synthetic data mode
```

This allows you to test your training loop, GPU utilization, and code logic without data I/O delays.

:::{tip} The synthetic data loader will be added to the main branch soon. It generates random data matching your dataset's dimensions, perfect for debugging before running full-scale experiments.
:::

### Option 2: Using Real Data

If you need to debug with actual data:

``` bash
# Copy data to fast local storage
cp /path/to/my/dataset.tar.gz $SLURM_TMPDIR/

# Extract (this can be slow for large files)
tar -xzf $SLURM_TMPDIR/dataset.tar.gz -C $SLURM_TMPDIR/

# Update your data path
export DATA_PATH="$SLURM_TMPDIR/dataset"
```

:::{warning} For large datasets, copying and unpacking to `$SLURM_TMPDIR` can take significant time. Use synthetic data for initial debugging, then test with real data once your code is working.
:::

## Debugging Your Code

Now you can run and debug your code interactively:

``` bash
# Test your training script
python ClimatExML/train.py --debug

# Run Python interactively
python
>>> import torch
>>> # debug as needed
```

## Exiting the Session

When you're done debugging:

``` bash
# Exit the interactive shell
exit

# This will release your allocation` 
```

The session will also automatically end when the time limit is reached.

## Canceling an Allocation

If you realize you requested the wrong resources or no longer need the allocation:

``` bash
# Cancel by job ID (find ID with squeue -u $USER)
scancel JOBID

# Cancel all your jobs
scancel -u $USER
```

:::{tip} **Best practice:** Request slightly more time than you think you'll need. If your session times out while debugging, you'll lose your work and need to request a new allocation.
:::

## Common Interactive Session Workflow

1.  Request allocation with  `salloc`
2.  Wait for email notification (or check with  `squeue -u $USER`)
3.  Connect with  `srun --pty bash`
4.  Load modules and activate environment
5.  Test and debug your code
6.  Exit when done

This workflow helps you catch errors before submitting long-running batch jobs.