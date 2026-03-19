#!/bin/bash
#SBATCH --job-name=sim_array
#SBATCH --output=logs/sim_%A_%a.out      # %A = job ID, %a = array task ID
#SBATCH -A m2616
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --error=logs/sim_%A_%a.err
#SBATCH --time=01:00:00                  # walltime per task — tune to your sim length
#SBATCH --mem=2G                         # memory per task
#SBATCH --cpus-per-task=1               # each task is single-threaded
#SBATCH --array=0-0                      # overridden at submission time by submit.py
#SBATCH --signal=SIGUSR1@180

# ---------------------------------------------------------------------------
# These variables are set by submit.py via --export when calling sbatch.
# Defaults here are fallbacks for manual testing.
# ---------------------------------------------------------------------------
N_PROC_PER_ENERGY=${N_PROC_PER_ENERGY:-20}
N_EVENTS_PER_PROC=${N_EVENTS_PER_PROC:-5000}
ENERGIES_FILE=${ENERGIES_FILE:-config/energies.txt}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/$USER/sim_output}
CONFIG_FILE=${CONFIG_FILE:-config/sim_config.yaml}

# ---------------------------------------------------------------------------
# Derive (energy_idx, proc_idx) from the flat task ID
# ---------------------------------------------------------------------------
ENERGY_IDX=$(( SLURM_ARRAY_TASK_ID / N_PROC_PER_ENERGY ))
PROC_IDX=$(( SLURM_ARRAY_TASK_ID % N_PROC_PER_ENERGY ))

# Read the energy value from the file (1-based line indexing for sed)
ENERGY=$(sed -n "$(( ENERGY_IDX + 1 ))p" "$ENERGIES_FILE")

if [[ -z "$ENERGY" ]]; then
    echo "ERROR: Could not read energy for ENERGY_IDX=$ENERGY_IDX from $ENERGIES_FILE"
    exit 1
fi

echo "========================================"
echo "SLURM_ARRAY_TASK_ID : $SLURM_ARRAY_TASK_ID"
echo "ENERGY_IDX          : $ENERGY_IDX"
echo "PROC_IDX            : $PROC_IDX"
echo "ENERGY              : $ENERGY GeV"
echo "N_EVENTS_PER_PROC   : $N_EVENTS_PER_PROC"
echo "OUTPUT_DIR          : $OUTPUT_DIR"
echo "========================================"

# Load your environment (adjust module names to your cluster)
# module load python/3.11
# module load root/6.28
# source /path/to/your/venv/bin/activate

mkdir -p logs

# chmod 644 $CONFIG_FILE
# podman-hpc run --rm --cfs --cvmfs --scratch -v $(pwd):/srv/ddsim caloxtreme:v0.2 \
#     "/srv/ddsim/scripts/ddsim_batch.py" \
#     --energy     "$ENERGY"          \
#     --proc-idx   "$PROC_IDX"        \
#     --n-events   "$N_EVENTS_PER_PROC" \
#     --output-dir "$OUTPUT_DIR"      \
#     --template     "$CONFIG_FILE"

podman-hpc run --rm --cfs --cvmfs --scratch -v $(pwd):/srv/ddsim caloxtreme:v0.2 \
    python /srv/ddsim/scripts/ddsim_batch.py \
    --energy     $ENERGY          \
    --proc-idx   $PROC_IDX        \
    --n-events   $N_EVENTS_PER_PROC \
    --output_dir $OUTPUT_DIR    \
    --template  $CONFIG_FILE

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
    echo "ERROR: ddsim_batch.py exited with code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "Task $SLURM_ARRAY_TASK_ID completed successfully."