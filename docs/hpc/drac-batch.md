# Batch Job Submission

**Tested on:** Nibi cluster  
**Last updated:** October 2025

Batch jobs allow you to submit long-running training tasks that execute without requiring an active connection. This is the primary way to run production training jobs on DRAC.

## Creating a Batch Submission Script

Create a file (e.g., `job_submit.slurm`) with your job configuration and execution commands:

```bash
#!/bin/bash

##############################
#       SLURM SETTINGS       #
##############################

#SBATCH --account=def-monahana           # PI allocation account
#SBATCH --mail-user=sbeairsto@uvic.ca    # Email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL       # When to send emails

#SBATCH --job-name=climatex_training     # Job name
#SBATCH --nodes=1                        # Number of nodes, usually 1
#SBATCH --ntasks=1                       # Number of tasks, usually 1
#SBATCH --cpus-per-task=8                # Number of CPUs
                                         # Ensure 'num_workers' in DataLoader <= this
#SBATCH --mem=80G                        # RAM, scales with batch size
#SBATCH --time=00-15:10:00               # Max runtime (DD-HH:MM:SS)
                                         # Cannot be extended after submission

#SBATCH --gres=gpu:h100:1                # Request 1 H100 GPU

##############################
#     JOB SETUP & LOGGING    #
##############################

# Define output log file using SLURM job ID
OUTPUT_FILE="$HOME/scratch/climatex/run_${SLURM_JOB_ID}.log"
mkdir -p "$(dirname "$OUTPUT_FILE")"

##############################
#    ENVIRONMENT SETUP       #
##############################

nvidia-smi
module purge
module load cuda/12.9

# Activate virtual environment
source $HOME/projects/def-monahana/$USER/ClimatEx/ClimatExML/climatexvenv/bin/activate
cd $HOME/projects/def-monahana/$USER/ClimatEx/ClimatExML

##############################
#       DATA TRANSFER        #
##############################

# Copy and extract dataset to $SLURM_TMPDIR
# Note: $SLURM_TMPDIR is fast local SSD storage

cp $HOME/path/to/my/data.tar.gz $SLURM_TMPDIR

# Set threads for parallel decompression
PIGZ_THREADS=${SLURM_CPUS_PER_TASK:-1}
echo "Using $PIGZ_THREADS threads for pigz decompression."

# Extract into SLURM_TMPDIR
mkdir -p "$SLURM_TMPDIR"
tar -C "$SLURM_TMPDIR" -I "pigz -d -p $PIGZ_THREADS" -xf "$SLURM_TMPDIR/data.tar.gz"

echo "Files extracted to $SLURM_TMPDIR/data"

# Verify extraction
find "$SLURM_TMPDIR" -maxdepth 3 -type d -print

##############################
#     RUN TRAINING SCRIPT    #
##############################

# Source any environment variables
. env_vars.sh

# Run training with output redirected to log file
python ClimatExML/train.py > "$OUTPUT_FILE" 2>&1

echo "Job completed at: $(date)"
```


### Submitting the Job

``` bash
sbatch job_submit.slurm
```

You'll receive a job ID confirmation:

```text
Submitted batch job 12345678
```

### Monitoring Your Job

``` bash
# Check job status
squeue -u $USER

# View detailed job information
scontrol show job JOBID

# Cancel a job if needed
scancel JOBID
```

:::{note} Wait times vary from minutes to hours depending on requested resources and cluster availability. You'll receive email notifications when the job begins, ends, or fails.
:::

### Job Outputs and Logs

SLURM automatically creates output files in the directory where you submitted the job:

-   **`slurm-JOBID.out`**: Standard output (stdout) and errors (stderr) from your job
-   **Custom log files**: Any files you specify (e.g.,  `$OUTPUT_FILE`  in the script above)

``` bash
# View job output while it's running
tail -f slurm-12345678.out

# View your custom log
tail -f $HOME/scratch/climatex/run_12345678.log
```

:::{tip} Redirect your training script's output to a custom log file in `/scratch/` so you can easily find and manage logs for different experiments. 
:::

### Key SLURM Parameters

| Parameter | Description | Notes | 
|-----------|-------------|-------| 
| `--account` | PI allocation account | Format: `def-piname` or `rrg-piname` |
 | `--time` | Maximum runtime | Format: `DD-HH:MM:SS`, cannot be extended | 
 | `--mem` | RAM allocation | Scale with batch size | 
 | `--cpus-per-task` | Number of CPUs | Should match DataLoader `num_workers` | 
 | `--gres` | GPU resources | `gpu:h100:1` for 1 H100 GPU on Nibi |
  | `--mail-type` | Email notifications | `BEGIN,END,FAIL` recommended |

### Best Practices

1.  **Always use  `$SLURM_TMPDIR`**  for data during training (5-10x faster I/O)
2.  **Request slightly more time than needed**  (jobs killed at time limit)
3.  **Match  `--cpus-per-task`  to DataLoader workers**  for optimal performance
4.  **Save checkpoints regularly**  to  `/scratch/`  in case of failures
5.  **Test with interactive sessions first**  before submitting long jobs

### Additional Resources

-   [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
-   [DRAC SLURM Guide](https://docs.alliancecan.ca/wiki/Running_jobs)
-   {doc}`drac-interactive`
-   {doc}`drac-optimization`