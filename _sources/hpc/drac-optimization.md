## Performance and Optimization 
**Tested on:** Nibi cluster (H100 GPUs)
**Last updated:** October 2025

Optimizing your training performance on DRAC ensures efficient use of compute resources and faster experiment iteration. This section covers key metrics to monitor and parameters to tune for H100 GPUs on Nibi.
 
 ### Optimization Workflow 
 :::{tip} **Use interactive sessions for optimization.** Request a GPU allocation (see [Interactive Sessions](interactive.md)) and use the synthetic data loader to quickly iterate on performance tuning without waiting for data transfer.
 ::: 
 ``` bash 
 # Request interactive session for optimization
 salloc --time=02:00:00 --mem=80G --cpus-per-task=8\
 --account=def-yourpi --gres=gpu:h100:1
 
 # Connect and setup
 srun --pty bash
 module purge
 module load cuda/12.9
 source /path/to/climatexvenv/bin/activate
 
 # Run training with synthetic data for quick testing
 python train.py --use_synthetic_data
 ```

## Key Performance Metrics

### GPU Memory Usage

**Target:** ~90% of available VRAM (72GB of 80GB on H100)

This metric shows how much GPU memory is allocated. Monitor this in Comet (see {doc}`tracking` for setup) under "GPU memory usage."

:::{warning} **Do not confuse with "GPU memory utilization"** - that measures how much of the allocated memory is actively being used at any given moment, not total allocation.
:::

**How to optimize:**

-   Increase batch size if usage is low (<70%)
-   Decrease batch size if hitting OOM errors
-   Use powers of 2 for batch sizes: 4, 8, 16, 32, 64, 96, 128...

**Recommended starting point:** Batch size of 96 works well for typical ClimatExML models.

### GPU Power Usage

**Target:** 550-650W average during training

Monitor power draw in Comet or with `nvidia-smi`. Lower power usage may indicate the GPU is not being fully utilized.

``` bash
# Monitor power in real-time during interactive session
watch -n 1 nvidia-smi
```

Look for the "Power Draw / Cap" column - you want to see consistent usage in the 550-650W range.

**Low power usage (<500W) may indicate:**

-   Data loading bottleneck (GPU waiting for data)
-   Insufficient batch size
-   Too few DataLoader workers

## Tuning Parameters

### Batch Size

Batch size has the largest impact on GPU memory usage and training speed.

**Finding optimal batch size:**

1.  Start with a power of 2 (e.g., 32)
2.  Monitor GPU memory usage in Comet or  `nvidia-smi`
3.  Increase incrementally: 32 → 64 → 96 → 128
4.  Stop when you reach ~90% memory usage or hit OOM

``` python
# In your training config
batch_size = 96  # Good starting point for H100
```

:::{note} **CRPS Loss Considerations:** If using CRPS-based loss functions with multiple realizations per batch member, you'll need to reduce batch size accordingly to stay within memory limits.
:::

:::{warning} Batch size increases are not linear with memory usage. Trial and error is necessary to find the sweet spot. 
:::

### Precision Mode
H100 GPUs are optimized for bfloat16 (bf16) computation and should use bf16 instead of the default mixed precision (fp16).

 ```yaml
  # In config.yaml
  precision: bf16-mixed # Use bf16, not 16-mixed (fp16)
 ```

See {doc}`customizing` for full configuration details.

**Benefits on H100:**

-   ~2x training speedup vs fp32
-   More stable than fp16 (wider dynamic range)
-   Optimized for H100 Tensor Cores

:::{warning} Default mixed precision uses fp16. Always explicitly set `bf16-mixed` for H100s.
:::


### Number of Workers
DataLoader workers handle data loading in parallel. Match this to your CPU allocation.

``` python
# In your DataLoader configuration
num_workers = 8  # Match --cpus-per-task in SLURM script
` ``

``` bash
# In your SLURM script
#SBATCH --cpus-per-task=8  # Should match num_workers` 
```

**Guidelines:**

-   Start with  `num_workers = cpus-per-task`
-   Too few workers: GPU waits for data (low power usage)
-   Too many workers: Overhead from context switching

### DataLoader Optimizations

Several PyTorch DataLoader options can significantly improve performance:
``` python
train_loader = DataLoader(
    dataset,
    batch_size=96,
    num_workers=8,
    pin_memory=True,           # Faster CPU-to-GPU transfer
    persistent_workers=True,    # Keep workers alive between epochs
    prefetch_factor=4,         # Pre-load 4 batches per worker
)
```

**Key parameters:**

-   **`pin_memory=True`**: Enables faster data transfer to GPU (recommended for CUDA)
-   **`persistent_workers=True`**: Avoids worker restart overhead between epochs
-   **`prefetch_factor=4`**: Number of batches each worker pre-loads

## Monitoring Performance

### During Training (Real-time)
**Option 1: Comet Dashboard**

Comet automatically tracks GPU metrics in real-time. See {doc}`tracking` for setup instructions.

Key metrics to watch:

-   GPU memory usage (target: ~90%)
-   GPU power draw (target: 550-650W)
-   Training throughput (samples/sec)

**Option 2: nvidia-smi (Interactive Sessions)**
```bash
# In an interactive session, SSH to your compute node
watch -n 1 nvidia-smi

# Look for:
# - Memory-Usage: should be ~72000MiB / 81920MiB (90%)
# - Power Draw: should be 550-650W
# - GPU-Util: should be high (>80%)
```

### After Training
``` bash
# Check job efficiency
seff JOBID

# Shows:
# - CPU efficiency
# - Memory efficiency 
# - Job duration` 
```

## Common Performance Issues

### Issue: Low GPU Utilization (<50%)

**Symptoms:**

-   Low power usage (<400W)
-   Low GPU utilization percentage
-   Slow training

**Solutions:**

1.  Increase batch size
2. Use `bf16-mixed` precision
3.  Increase  `num_workers`  in DataLoader
4.  Enable DataLoader optimizations (`pin_memory`,  `persistent_workers`)
5.  Verify data is in  `$SLURM_TMPDIR`  (not reading from network storage)

### Issue: GPU Memory OOM

**Symptoms:**

``` csharp
`RuntimeError: CUDA out of memory` 
``` 
**Solutions:**

1.  Reduce batch size

### Issue: Data Loading Bottleneck

**Symptoms:**

-   Low GPU utilization despite reasonable batch size
-   High CPU usage
-   Training speed doesn't improve with more workers

**Solutions:**

1.  Verify data is in  `$SLURM_TMPDIR`, not network storage
2.  Use synthetic data loader to test if data I/O is the bottleneck

## Optimal Configuration for Nibi H100s

Based on testing, these settings work well for typical ClimatExML training:

``` bash
`# SLURM settings
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G` 
```

``` python
# DataLoader settings
DataLoader(
    dataset,
    batch_size=96,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
```

**Expected performance:**

-   GPU memory usage: ~72GB / 80GB (90%)
-   Power draw: 550-650W average
-   GPU utilization: >80%

:::{tip} These are starting points. Your optimal settings may vary depending on model architecture, input data size, and loss function. Always benchmark with your specific configuration.
:::

### Benchmarking Checklist

Before running long training jobs:

-   [ ] Test with synthetic data loader in interactive session
-   [ ] Verify GPU memory usage is ~90%
-   [ ] Check power draw is 550-650W
-   [ ] Confirm no data loading bottlenecks
-   [ ] Validate DataLoader settings match CPU allocation
-   [ ] Monitor first few epochs in Comet dashboard

## Additional Resources

-   {doc}`../trackingtracking`  - Setting up Comet for metric tracking
-   {doc}`drac-interactive`  - Using interactive sessions for optimization
-   [PyTorch DataLoader Documentation](https://pytorch.org/docs/stable/data.html)
-   [NVIDIA GPU Performance Guide](https://docs.nvidia.com/deeplearning/performance/)