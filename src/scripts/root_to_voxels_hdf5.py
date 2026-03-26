from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import uproot
from tqdm.rich import tqdm

from src.batch.submit_range import get_parser
from src.scripts.root_utils import (
    BARREL_KEY,
    PARTICLE_KEY,
    compute_relative_position,
    extract_calo_showers,
    preprocess_calo_showers,
    preprocess_particles,
)
from src.scripts.voxelize import digitize_shower, get_voxels

SCALE_FACTOR=0.0266
WRITE_BATCH_SIZE = 100

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Convert ROOT calorimeter showers to voxelized HDF5 datasets.")
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to a ROOT file or a directory containing ROOT files.",
    )
    parser.add_argument(
        "--binning-xml",
        required=True,
        type=Path,
        help="Path to voxel binning XML file.",
    )
    parser.add_argument(
        "--envelope-xml",
        required=True,
        type=Path,
        help="Path to detector envelope XML file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output HDF5 file path.",
    )
    parser.add_argument(
        "--tree-name",
        default="events",
        help="ROOT tree name (default: events).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (mp.cpu_count() or 1)),
        help="Number of worker processes for per-event voxelization.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return get_parser().parse_args()

def collect_root_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix == ".root" else []
    if input_path.is_dir():
        return sorted(input_path.rglob("*.root"))
    return []


def event_to_voxel_shower(
    event_data,
    envelope_xml: str,
    binning_xml: str,
) -> np.ndarray | None:
    barrel_keys = [key for key in event_data.fields if key.startswith(f"{BARREL_KEY}.")]
    particle_keys = [
        key for key in event_data.fields if key.startswith(f"{PARTICLE_KEY}.")
    ]

    barrel_df = pd.DataFrame(event_data[barrel_keys].to_list())
    particle_df = pd.DataFrame(event_data[particle_keys].to_list())

    if len(particle_df) == 0:
        return None

    barrel_df = barrel_df.rename(
        columns=lambda col: col.replace(".", "_").split("_")[-1]
    )
    particle_df = particle_df.rename(columns=lambda col: col.replace(".", "_"))
    particle_df = particle_df[
        particle_df["MCParticles_generatorStatus"] == 1
    ].reset_index(drop=True)
    assert len(particle_df) == 1, "Expected exactly one primary particle per event."
    momentum = particle_df[
        ["MCParticles_momentum_x", "MCParticles_momentum_y", "MCParticles_momentum_z"]
    ].to_numpy()
    momentum = np.linalg.norm(momentum, axis=1) * 1000  # convert to MeV

    barrel_df = compute_relative_position(barrel_df, particle_df)
    voxels = get_voxels(particle_df, envelope_xml, binning_xml)

    original_voxels = voxels.copy()
    voxels = voxels / 1000.0
    unitless_cols = [
        "z_bin_index",
        "r_bin_index",
        "phi_bin_index",
        "phi_bin_centre",
        "phi_bin_min",
        "phi_bin_max",
        "layer_id",
    ]
    voxels[unitless_cols] = original_voxels[unitless_cols]

    _, energized_voxels = digitize_shower(barrel_df, voxels)
    energized_voxels = energized_voxels.sort_values(
        by=["z_bin_index", "r_bin_index", "phi_bin_index"]
    ).reset_index(drop=True)
    return energized_voxels["binned_energy"].to_numpy(dtype=np.float64), momentum

def scale_showers(voxels: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Scale the energy values in the voxel array by a given factor.

    Parameters:
    - voxels: A numpy array of shape (N, 4) where the first three columns are bin indices and the fourth column is energy.
    - scale_factor: The factor by which to scale the energy values.

    Returns:
    - A new numpy array with the same shape as `voxels` but with scaled energy values.
    """
    
    return voxels / scale_factor

def _voxelize_event_data(event_data, envelope_xml: str, binning_xml: str) -> np.ndarray | None:
    """Worker function that receives event data explicitly."""
    return event_to_voxel_shower(event_data, envelope_xml=envelope_xml, binning_xml=binning_xml)


def flush_to_hdf5(
    h5_file: h5py.File,
    showers_array: list[np.ndarray],
    incident_array: list[np.ndarray],
) -> int:
    """Flush buffered showers to HDF5 in append mode and clear buffers."""
    if len(showers_array) == 0:
        return 0

    if "showers" in h5_file:
        shower_ds = h5_file["showers"]
        energy_ds = h5_file["incident_energies"]

        if shower_ds.shape[1] != showers_array.shape[1]:
            raise ValueError(
                "Cannot append showers with different voxel length: "
                f"existing {shower_ds.shape[1]}, new {showers_array.shape[1]}"
            )

        start = shower_ds.shape[0]
        n_new = showers_array.shape[0]

        shower_ds.resize((start + n_new, shower_ds.shape[1]))
        shower_ds[start : start + n_new] = showers_array

        energy_ds.resize((start + n_new, 1))
        energy_ds[start : start + n_new] = incident_array
    else:
        h5_file.create_dataset(
            "showers",
            data=showers_array,
            maxshape=(None, showers_array.shape[1]),
            compression="gzip",
            chunks=showers_array.shape,
        )
        h5_file.create_dataset(
            "incident_energies",
            data=incident_array,
            maxshape=(None, 1),
            compression="gzip",
            chunks=incident_array.shape,
        )

    written = len(showers_array)

    return written


def main() -> int:
    args = parse_args()

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    root_files = collect_root_files(args.input)
    if not root_files:
        raise FileNotFoundError(f"No ROOT files found at: {args.input}")

    logging.info(f"Found {len(root_files)} ROOT files to process.")

    expected_num_voxels: int | None = None
    written_this_run = 0

    # Determine if parallel processing is possible (once, not per-file)
    use_parallel = args.num_workers > 1
    if use_parallel:
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context()

        if ctx.get_start_method() != "fork":
            use_parallel = False

    # Create pool once before processing all files
    pool = None
    if use_parallel:
        ctx = mp.get_context("fork")
        pool = ctx.Pool(processes=args.num_workers)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        logging.warning(
            "Output file %s already exists and will be overwritten.",
            args.output,
        )
        os.remove(args.output)

    with h5py.File(args.output, "w") as h5_file:

        try:
            for root_file in tqdm(root_files, desc="Processing ROOT files"):
                for array in uproot.iterate(
                    f"{str(root_file)}:{args.tree_name}",
                    library="ak",
                    step_size=WRITE_BATCH_SIZE,
                ):
                    array = preprocess_calo_showers(array)
                    array = preprocess_particles(array)

                    if len(array) == 0:
                        continue

                    file_use_parallel = use_parallel and len(array) > 1

                    showers = []
                    incident_energies = []

                    if file_use_parallel and pool is not None:
                        tasks = [
                            (array[i], str(args.envelope_xml), str(args.binning_xml))
                            for i in range(len(array))
                        ]
                        shower_iter = pool.starmap(_voxelize_event_data, tasks)

                        for shower in shower_iter:
                            if shower is None:
                                continue

                            voxel_count = len(shower[0])
                            if expected_num_voxels is None:
                                expected_num_voxels = voxel_count
                            elif voxel_count != expected_num_voxels:
                                raise ValueError(
                                    "Inconsistent voxel vector size across events: "
                                    + (
                                        f"expected {expected_num_voxels}, "
                                        f"got {voxel_count} in {root_file}."
                                    )
                                )

                            showers.append(scale_showers(shower[0], SCALE_FACTOR))
                            incident_energies.append(shower[1])

                    else:
                        for event_idx in range(len(array)):
                            shower = event_to_voxel_shower(
                                array[event_idx],
                                envelope_xml=str(args.envelope_xml),
                                binning_xml=str(args.binning_xml),
                            )
                            if shower is None:
                                continue

                            voxel_count = len(shower[0])
                            if expected_num_voxels is None:
                                expected_num_voxels = voxel_count
                            elif voxel_count != expected_num_voxels:
                                raise ValueError(
                                    "Inconsistent voxel vector size across events: "
                                    + (
                                        f"expected {expected_num_voxels}, "
                                        f"got {voxel_count} in {root_file}."
                                    )
                                )

                            showers.append(scale_showers(shower[0], SCALE_FACTOR))
                            incident_energies.append(shower[1])

                    
                    showers_array = np.stack(showers, axis=0).astype(np.float32)
                    incident_array = np.array(incident_energies).reshape(-1, 1).astype(np.float32)

                    written_this_run += flush_to_hdf5(h5_file, showers_array, incident_array)

            # Flush any remaining buffered events
            # written_this_run += flush_to_hdf5(h5_file, showers_array, incident_array)
        finally:
            # Clean up pool after processing all files
            if pool is not None:
                pool.close()
                pool.join()

        if written_this_run == 0:
            raise RuntimeError(
                "No valid showers were produced from the provided ROOT input."
            )

        total_showers = h5_file["showers"].shape[0]
        num_voxels = h5_file["showers"].shape[1]

    print(
        f"Saved {written_this_run} new showers "
        f"(total in file: {total_showers}) with {num_voxels} voxels each to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
