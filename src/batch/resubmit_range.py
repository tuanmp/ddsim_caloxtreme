#!/usr/bin/env python3
"""
resubmit.py
-----------
Scans the output directory for missing or invalid .root files, maps them
back to Slurm array task IDs, and resubmits only those tasks.

Usage:
    python -m src.batch.resubmit_range [--dry-run]
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import yaml

from src.batch.submit_range import (
    MAX_CONCURRENT,
    MEMORY,
    N_EVENTS_PER_PROC,
    N_EVENTS_TARGET,
    SLURM_SCRIPT,
    assemble_cmd,
    get_parser,
)
from src.scripts.check_valid import is_valid_root_file
from src.scripts.utils.config import fill_config, get_output_subdir


def find_missing_task_ids(check_dir, n_proc):
    missing = []
    for proc_idx in range(n_proc):
        outfile = os.path.join(check_dir, f"edm4hep_proc_{proc_idx}.root")
        if not is_valid_root_file(outfile):
            missing.append(proc_idx)
            print(f"  MISSING task {proc_idx:4d} -> proc={proc_idx}  ({outfile})")
            
            if os.path.exists(outfile):
                os.remove(outfile)  # Remove invalid output before resubmission.
    return missing


def main():

    parser = get_parser()
    args = parser.parse_args()

    # Reuse the same config file shape expected by fill_config.
    setattr(args, "template", args.config_file)
    args = fill_config(args)

    check_dir = get_output_subdir(args)

    n_tasks = args.n_target_events // args.n_event_per_proc


    print(f"Scanning {check_dir} for missing/invalid output files...")
    missing = find_missing_task_ids(check_dir, n_tasks)

    if not missing:
        print("All output files present and valid. Nothing to resubmit.")
        return

    print(f"\n{len(missing)} task(s) to resubmit.")

    # Build a compact Slurm array spec from missing task IDs.
    array_spec = ids_to_array_spec(missing, args.max_concurrent)
    print(f"Array spec: {array_spec}")

    # Convert host path to the in-container path used by the batch runner.
    relative_config_path = args.config_file.resolve().relative_to(Path.cwd())
    config_path_in_container = os.path.join("/srv/ddsim", relative_config_path.as_posix())

    cmd = assemble_cmd(args, array_spec, config_path_in_container)

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