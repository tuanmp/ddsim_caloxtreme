import argparse
import logging
import os
import re
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SLURM_SCRIPT = "src/batch/voxelize_array.sh"
from src.batch.submit_range import MAX_CONCURRENT
from src.scripts.root_to_voxels_hdf5 import collect_root_files
from src.scripts.root_to_voxels_hdf5 import get_parser as _get_parser


def get_parser():
    parser = _get_parser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the sbatch command without running it")
    parser.add_argument("--walltime", default="01:00:00", type=str,
                        help="Walltime for each job (default: 01:00:00)")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT,
                        help=f"Max concurrent tasks to throttle (default: {MAX_CONCURRENT})")
    return parser

def assemble_cmd(args, array_spec, all_input_files, all_output_files):

    return  [
        "sbatch",
        f"--array={array_spec}",
        f"--time={args.walltime}",
        f"--cpus-per-task={args.num_workers}",
        f"--export=ALL," \
        f'BINNING_XML={args.binning_xml},' \
        f'ENVELOPE_XML={args.envelope_xml},' \
        f'ALL_INPUT_FILES="{all_input_files}",' \
        f'ALL_OUTPUT_FILES="{all_output_files}",' \
        f'TREE_NAME={args.tree_name},' \
        f'NUM_WORKERS={args.num_workers}', 
        SLURM_SCRIPT
    ]

def main():

    args = get_parser().parse_args()

    root_files = collect_root_files(args.input)

    if not root_files:
        logging.error("No valid ROOT files found.")
        return 1
    logging.info(f"Found {len(root_files)} ROOT files to process.")

    file_to_process = []

    for root_file in root_files:

        output_file = args.output / (root_file.stem + ".hdf5")

        if not output_file.exists():
            file_to_process.append((root_file, output_file))

    n_tasks = len(file_to_process)
    if n_tasks == 0:
        logging.info("All files have already been processed. Nothing to do.")
        return 0
    
    logging.info(f"{n_tasks} files need to be processed.")
    array_spec = f"0-{n_tasks-1}%{args.max_concurrent}"  # throttle to 10 concurrent jobs

    all_input_files = " ".join(str(f[0]) for f in file_to_process).lstrip()
    all_output_files = " ".join(str(f[1]) for f in file_to_process).lstrip()

    cmd = assemble_cmd(args, array_spec, all_input_files, all_output_files)

    logging.info("Command:")
    logging.info("  " + " ".join(cmd).lstrip()[:200] + " ...")  # Truncate long command for logging

    if args.dry_run:
        logging.info("[dry-run] Not submitting.")
        return 0
    
    os.makedirs("logs", exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    logging.info(result.stdout)
    if result.returncode != 0:
        logging.error(result.stderr)
        return result.returncode
    return 0


if __name__ == "__main__":
    main()









