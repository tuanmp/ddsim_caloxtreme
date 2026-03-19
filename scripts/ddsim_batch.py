"""
Given
- a template yaml config file
- a csv file with columns of incident energy needed for steering
The script does the following:

1. Create an output directory if it doesn't exist
2. Read the csv file and extract the incident energy values
3. For each incident energy, create a subdirectory if it doesn't exist and generate a yaml config file by replacing the placeholder in the template with the actual energy value
4. Check if enough events have been generated for each energy value, and if not, add the missing energy values to a list for further processing

"""


import logging
import os
import shutil
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml
from check_valid import is_valid_root_file
from ddsim_run import run_ddsim
from utils.app_logging import setup_logging


def parse_args():
    parser = ArgumentParser(description="Batch steering script for generating yaml config files based on incident energy values from a csv file.")
    parser.add_argument("--template", required=True, help="Path to the template yaml config file.")
    parser.add_argument("--energy", type=float, required=True, help="Path to the csv file containing incident energy values.")
    parser.add_argument("--output_dir", required=True, help="Directory to store the generated yaml config files and output data.")
    parser.add_argument("--proc-idx", type=int, default=0, help="Process index for parallel execution.")
    parser.add_argument("--n-events", type=int, default=1000, help="Number of events to simulate per process.")

    return parser.parse_args()

SUBDIR = "energy"

def fill_config(args):

    with open(args.template, 'r') as f:
        template_config = yaml.safe_load(f)
    
    for key, value in template_config.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    setattr(args, "gun_energy", float(args.energy))
    setattr(args, "seed", hash((args.energy, args.proc_idx)) % (2**32))  # Generate a unique seed based on energy and process index
    
    return args

def run_sim(config, output_file_path):
    # Simulate a single event using the provided config and move the output to the specified path

    uid = uuid4()  # Generate a unique identifier for the simulation run
    temp_output_path = f"/tmp/{uid}.root"  # Generate a unique temporary output file path

    run_ddsim(input_path=None, output_path=temp_output_path, config=config, logger=setup_logging(str(uid), logging.ERROR))
    # Assuming the output file is generated at a known location, move it to the desired path

    time.sleep(1)  # Ensure the file is fully written before moving

    shutil.move(temp_output_path, output_file_path)

def main():
    # Create output directory if it doesn't exist

    # Read the csv file and extract incident energy values
    args = parse_args()

    config = fill_config(args)

    output_dir = Path(args.output_dir) / config.dataset / f"{SUBDIR}_{int(config.gun_energy)}_GeV"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"edm4hep_proc_{args.proc_idx}.root"

    run_sim(config, output_path)

    valid = is_valid_root_file(output_path)
    if valid:
        logging.info(f"Successfully generated valid output file: {output_path}")

        sys.exit(0)  # Exit with code 0 for success
    else:
        logging.error(f"Failed to generate valid output file: {output_path}")
        sys.exit(1)  # Exit with code 1 for failure

if __name__ == "__main__":
    main()