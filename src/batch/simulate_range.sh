#!/bin/bash
#SBATCH --job-name=sim_array
#SBATCH --output=logs/sim_range/sim_%A_%a.out      # %A = job ID, %a = array task ID
#SBATCH -A m2616
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --error=logs/sim_range/sim_%A_%a.err
#SBATCH --time=01:00:00                  # walltime per task — tune to your sim length
#SBATCH --cpus-per-task=1               # each task is single-threaded
#SBATCH --array=0-0                      # overridden at submission time by submit.py
#SBATCH --signal=SIGUSR1@180

# ---------------------------------------------------------------------------
# These variables are set by submit.py via --export when calling sbatch.
# Defaults here are fallbacks for manual testing.
# ---------------------------------------------------------------------------
N_EVENTS_PER_PROC=${N_EVENTS_PER_PROC:-5000}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/$USER/sim_output}
CONFIG_FILE=${CONFIG_FILE:-config/sim_config.yaml}


echo "========================================"
echo "SLURM_ARRAY_TASK_ID : $SLURM_ARRAY_TASK_ID"
echo "PROC_IDX            : $SLURM_ARRAY_TASK_ID"
echo "N_EVENTS_PER_PROC   : $N_EVENTS_PER_PROC"
echo "OUTPUT_DIR          : $OUTPUT_DIR"
echo "========================================"

# Load your environment (adjust module names to your cluster)
# module load python/3.11
# module load root/6.28
# source /path/to/your/venv/bin/activate

mkdir -p logs/sim_range/

podman-hpc run --rm --cfs --cvmfs --scratch -w /srv/ddsim -v $(pwd):/srv/ddsim caloxtreme:v0.2 \
    python -m src.scripts.ddsim_batch \
    --proc-idx   $SLURM_ARRAY_TASK_ID        \
    --events   $N_EVENTS_PER_PROC \
    --output_dir $OUTPUT_DIR    \
    --template  $CONFIG_FILE

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
    echo "ERROR: ddsim_batch.py exited with code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "Task $SLURM_ARRAY_TASK_ID completed successfully."