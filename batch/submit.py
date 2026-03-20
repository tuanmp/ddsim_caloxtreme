#!/usr/bin/env python3
"""
submit.py
---------
Computes the job array size from your physics parameters and submits the
Slurm job array. Run this once to launch the full campaign.

Usage:
    python scripts/submit.py [--dry-run]
"""

import argparse
import math
import os
import subprocess
import sys

# ===========================================================================
# CONFIGURE THESE
# ===========================================================================
ENERGIES_FILE    = "config/energies.txt"
CONFIG_FILE      = "config/sim_config.yaml"
SCRIPTS_DIR      = "scripts"
SLURM_SCRIPT     = "batch/simulate.sh"
OUTPUT_DIR  = "" #f"/scratch/{os.environ.get('USER', 'user')}/sim_output"

N_EVENTS_TARGET  = 100_000   # desired events per energy value
N_EVENTS_PER_PROC = 10000    # events produced per simulation process

# Slurm settings
MAX_CONCURRENT   = 50        # max tasks running simultaneously (throttle)
WALLTIME         = "03:30:00"
MEMORY           = "2G"
# ===========================================================================


def read_energies(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the sbatch command without running it")
    parser.add_argument("--energies-file", default=ENERGIES_FILE,
                        help="Path to the file containing energy values (default: config/energies.txt)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Directory to store simulation outputs (default: /scratch/$USER/sim_output)")
    parser.add_argument("--config-file", default=CONFIG_FILE,
                        help="Path to the yaml config file for simulations (default: config/sim_config.yaml)")
    args = parser.parse_args()

    energies = read_energies(args.energies_file)
    n_energies = len(energies)
    n_proc_per_energy = math.ceil(N_EVENTS_TARGET / N_EVENTS_PER_PROC)
    n_tasks = n_energies * n_proc_per_energy

    print(f"Energies        : {n_energies}  ({', '.join(energies)})")
    print(f"Processes/energy: {n_proc_per_energy}  ({N_EVENTS_PER_PROC} events each)")
    print(f"Total tasks     : {n_tasks}  (array 0-{n_tasks-1})")
    print(f"Max concurrent  : {MAX_CONCURRENT}")
    print(f"Output dir      : {args.output_dir}")

    array_spec = f"0-{n_tasks - 1}%{MAX_CONCURRENT}"

    cmd = [
        "sbatch",
        f"--array={array_spec}",
        f"--time={WALLTIME}",
        f"--mem={MEMORY}",
        f"--export=ALL," \
        f"N_PROC_PER_ENERGY={n_proc_per_energy}," \
        f"N_EVENTS_PER_PROC={N_EVENTS_PER_PROC}," \
        f"ENERGIES_FILE={args.energies_file}," \
        f"OUTPUT_DIR={args.output_dir}," \
        f"CONFIG_FILE={args.config_file}",
        SLURM_SCRIPT,
    ]

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
