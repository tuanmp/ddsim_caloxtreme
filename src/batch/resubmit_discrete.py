#!/usr/bin/env python3
"""
resubmit.py
-----------
Scans the output directory for missing or invalid .root files, maps them
back to Slurm array task IDs, and resubmits only those tasks.

Usage:
    python -m src.batch.resubmit_discrete [--dry-run]
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import yaml

# Support running this file directly (e.g. `uv run src/batch/resubmit_discrete.py`).
# In that mode Python does not add the repository root to sys.path automatically.
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
        
from src.batch.resubmit_range import find_missing_task_ids
from src.batch.submit_discrete import (
    MAX_CONCURRENT,
    MEMORY,
    N_EVENTS_PER_PROC,
    N_EVENTS_TARGET,
    SLURM_SCRIPT,
    WALLTIME,
    assemble_cmd,
    get_parser,
    read_energies,
)
from src.scripts.check_valid import is_valid_root_file
from src.scripts.utils.config import fill_config, get_output_subdir


def main():

    parser = get_parser()
    args = parser.parse_args()

    energies = read_energies(args.energies_file)
    n_proc_per_energy = math.ceil(args.n_target_events / args.n_event_per_proc)

    # Reuse the same config schema expected by fill_config.
    setattr(args, "template", args.config_file)

    relative_config_path = args.config_file.resolve().relative_to(Path.cwd())
    config_path_in_container = os.path.join("/srv/ddsim", relative_config_path.as_posix())
    
    missing = []
    for en_idx, energy in enumerate(energies):
        setattr(args, "energy", energy)
        # Expand config for the current energy point.
        args = fill_config(args)
        check_dir = get_output_subdir(args)


        print(f"Scanning {check_dir} for missing/invalid output files...")
        missing_from_this_energy = find_missing_task_ids(check_dir, n_proc_per_energy)
        # Map per-energy process indices to global Slurm array task IDs.
        missing.extend([
            en_idx * n_proc_per_energy + proc_idx
            for proc_idx in missing_from_this_energy
        ])

        if not missing:
            print("All output files present and valid. Nothing to resubmit.")
        else:
            print(f"\n{len(missing)} task(s) to resubmit.")
            continue

    # Build a compact Slurm array spec from missing task IDs.
    array_spec = ids_to_array_spec(missing, MAX_CONCURRENT)
    print(f"Array spec: {array_spec}")

    cmd = assemble_cmd(args, array_spec, n_proc_per_energy, config_path_in_container)

    print("\nCommand:")
    print("  " + " ".join(cmd))

    if args.dry_run:
        print("\n[dry-run] Not submitting.")
        return

    if len(missing) == 0:
        print("\nNo tasks to resubmit. Exiting.")
        return
    os.makedirs("logs", exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr, file=sys.stderr)
        sys.exit(result.returncode)


def ids_to_array_spec(ids: list[int], max_concurrent: int) -> str:
    """Compress a list of integers into a compact Slurm array range string."""
    if not ids:
        return ""
    ids = sorted(set(ids))
    ranges = []
    start = end = ids[0]
    for i in ids[1:]:
        if i == end + 1:
            end = i
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = i
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(ranges) + f"%{max_concurrent}"


if __name__ == "__main__":
    main()