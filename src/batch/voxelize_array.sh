#!/bin/bash
#SBATCH --job-name=voxelize_array
#SBATCH --output=logs/voxelize/voxelize_%A_%a.out      # %A = job ID, %a = array task ID
#SBATCH -A m2616
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --error=logs/voxelize/voxelize_%A_%a.err
#SBATCH --time=01:00:00                  # walltime per task — tune to your sim length
#SBATCH --cpus-per-task=1               # overriden by --num-workers in the script, but set to 1 here since each worker is single-threaded
#SBATCH --array=0-0                      # overridden at submission time by submit.py
#SBATCH --signal=SIGUSR1@180

# replace quotation marks if any to avoid issues with passing arrays as environment variables
ALL_INPUT_FILES=${ALL_INPUT_FILES//\"/}
ALL_OUTPUT_FILES=${ALL_OUTPUT_FILES//\"/}

# echo $ALL_INPUT_FILES
# echo $ALL_OUTPUT_FILES

# number of arrays is always equal to the number of input files,
# so we can derive the input file for this task from the array ID
read -a INPUT_FILES <<< $ALL_INPUT_FILES   # convert space-separated string to array
read -a OUTPUT_FILES <<< $ALL_OUTPUT_FILES # convert space-separated string to array
INPUT_FILE=${INPUT_FILES[$SLURM_ARRAY_TASK_ID]}
OUTPUT_FILE=${OUTPUT_FILES[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "SLURM_ARRAY_TASK_ID : $SLURM_ARRAY_TASK_ID"
echo "INPUT_FILE          : $INPUT_FILE"
echo "OUTPUT_FILE         : $OUTPUT_FILE"
echo "CPUS_PER_TASK       : $SLURM_CPUS_PER_TASK"
echo "BINNING_XML         : $BINNING_XML"
echo "ENVELOPE_XML        : $ENVELOPE_XML"
echo "TREE_NAME           : $TREE_NAME"
echo "NUM_WORKERS         : $NUM_WORKERS"
echo "========================================"

# Load your environment (adjust module names to your cluster)
# module load python/3.11
# module load root/6.28
# source /path/to/your/venv/bin/activate

mkdir -p logs/voxelize/

module load python 

cmd="uv run -m src.scripts.root_to_voxels_hdf5 \
    --input  $INPUT_FILE  \
    --output $OUTPUT_FILE \
    --binning-xml $BINNING_XML \
    --envelope-xml $ENVELOPE_XML \
    --num-workers $SLURM_CPUS_PER_TASK \
    --tree-name $TREE_NAME"

echo $cmd

$cmd

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
    echo "ERROR: root_to_voxels_hdf5.py exited with code $EXIT_CODE"
    if [ -f $OUTPUT_FILE ]; then
        echo "Removing incomplete output file: $OUTPUT_FILE"
        rm -f "$OUTPUT_FILE"
    fi
    exit $EXIT_CODE
fi

echo "Task $SLURM_ARRAY_TASK_ID completed successfully."