#!/usr/bin/env python3
"""
submit.py
---------
Computes the job array size from your physics parameters and submits the
Slurm job array. Run this once to launch the full campaign.

Usage:
    python -m src.batch.submit_discrete [--dry-run]
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

# Submission defaults
ENERGIES_FILE    = "config/energies.txt"
CONFIG_FILE      = "config/sim_config.yaml"
SCRIPTS_DIR      = "scripts"
SLURM_SCRIPT     = "src/batch/simulate_discrete.sh"
OUTPUT_DIR  = ""

N_EVENTS_TARGET  = 100_000   # desired events per energy value
N_EVENTS_PER_PROC = 10000    # events produced per simulation process

# Slurm settings
MAX_CONCURRENT   = 400       # max tasks running simultaneously (throttle)
WALLTIME         = "01:45:00"
MEMORY           = "4G"


def read_energies(path):
    # Ignore empty lines and comment lines.
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the sbatch command without running it")
    parser.add_argument("--energies-file", default=ENERGIES_FILE,
                        help="Path to the file containing energy values (default: config/energies.txt)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path,
                        help="Directory to store simulation outputs (default: /scratch/$USER/sim_output)")
    parser.add_argument("--n-target-events", type=int, default=N_EVENTS_TARGET, 
                        help=f"Number of events to simulate per energy value (default: {N_EVENTS_TARGET})")
    parser.add_argument("--n-event-per-proc", type=int, default=N_EVENTS_PER_PROC,
                        help=f"Number of events to simulate per process (default: {N_EVENTS_PER_PROC})")
    parser.add_argument("--config-file", default=CONFIG_FILE, type=Path,
                        help="Path to the yaml config file for simulations (default: config/sim_config.yaml)")
    parser.add_argument("--walltime", default=WALLTIME, type=str,
                        help=f"Walltime for each job (default: {WALLTIME})")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT,
                        help=f"Max concurrent tasks to throttle (default: {MAX_CONCURRENT})")
    parser.add_argument("--memory", default=MEMORY, type=str,
                        help=f"Memory for each job (default: {MEMORY})")
    return parser


def assemble_cmd(args, array_spec, n_proc_per_energy, config_path):
    cmd = [
        "sbatch",
        f"--array={array_spec}",
        f"--time={args.walltime}",
        f"--mem={args.memory}",
        f"--export=ALL," \
        f"N_PROC_PER_ENERGY={n_proc_per_energy}," \
        f"N_EVENTS_PER_PROC={args.n_event_per_proc}," \
        f"ENERGIES_FILE={args.energies_file}," \
        f"OUTPUT_DIR={args.output_dir}," \
        f"CONFIG_FILE={config_path}",
        SLURM_SCRIPT,
    ]
    return cmd

def main():
    parser = get_parser()
    args = parser.parse_args()

    energies = read_energies(args.energies_file)
    n_energies = len(energies)
    n_proc_per_energy = math.ceil(args.n_target_events / args.n_event_per_proc)
    # One contiguous array spans all energies and all processes per energy.
    n_tasks = n_energies * n_proc_per_energy

    print(f"Energies        : {n_energies}  ({', '.join(energies)})")
    print(f"Processes/energy: {n_proc_per_energy}  ({args.n_event_per_proc} events each)")
    print(f"Total tasks     : {n_tasks}  (array 0-{n_tasks-1})")
    print(f"Max concurrent  : {args.max_concurrent}")
    print(f"Output dir      : {args.output_dir}")

    array_spec = f"0-{n_tasks - 1}%{args.max_concurrent}"

    # Convert host path to the in-container path used by the batch runner.
    relative_config_path = args.config_file.resolve().relative_to(Path.cwd())
    config_path_in_container = os.path.join("/srv/ddsim", relative_config_path.as_posix())

    cmd = assemble_cmd(args, array_spec, n_proc_per_energy, config_path_in_container)

    print("\nCommand:")
    print("  " + " ".join(cmd))

    if args.dry_run:
        print("\n[dry-run] Not submitting.")
        return

    os.makedirs("logs", exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr, file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
