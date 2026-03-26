from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm.rich import tqdm

from src.scripts.root_utils import (
    BARREL_KEY,
    PARTICLE_KEY,
    compute_relative_position,
    extract_calo_showers,
    preprocess_calo_showers,
    preprocess_particles,
)
from src.scripts.voxelize import digitize_shower, get_voxels

SCALE_FACTOR = 0.0266
WRITE_CHUNK_SIZE = 2048

_WORKER_ARRAY = None
_WORKER_ENVELOPE_XML: str | None = None
_WORKER_BINNING_XML: str | None = None
_WORKER_BARREL_KEYS: list[str] | None = None
_WORKER_PARTICLE_KEYS: list[str] | None = None

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Convert ROOT calorimeter showers to voxelized HDF5 datasets (optimized).")
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
    return parser.parse_args()


def collect_root_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix == ".root" else []
    if input_path.is_dir():
        return sorted(input_path.rglob("*.root"))
    return []


def _event_to_voxel_shower(
    event_data,
    envelope_xml: str,
    binning_xml: str,
    barrel_keys: list[str],
    particle_keys: list[str],
) -> tuple[np.ndarray, float] | None:
    barrel_df = pd.DataFrame(event_data[barrel_keys].to_list())
    particle_df = pd.DataFrame(event_data[particle_keys].to_list())

    if len(particle_df) == 0:
        return None

    barrel_df = barrel_df.rename(columns=lambda col: col.replace(".", "_").split("_")[-1])
    particle_df = particle_df.rename(columns=lambda col: col.replace(".", "_"))
    particle_df = particle_df[
        particle_df["MCParticles_generatorStatus"] == 1
    ].reset_index(drop=True)

    if len(particle_df) != 1:
        raise ValueError("Expected exactly one primary particle per event.")

    momentum_vec = particle_df[
        ["MCParticles_momentum_x", "MCParticles_momentum_y", "MCParticles_momentum_z"]
    ].to_numpy(dtype=np.float64)
    momentum_mev = float(np.linalg.norm(momentum_vec, axis=1)[0] * 1000.0)

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

    shower = energized_voxels["binned_energy"].to_numpy(dtype=np.float32) / SCALE_FACTOR
    return shower, momentum_mev


def _init_worker(
    array,
    envelope_xml: str,
    binning_xml: str,
    barrel_keys: list[str],
    particle_keys: list[str],
) -> None:
    globals()["_WORKER_ARRAY"] = array
    globals()["_WORKER_ENVELOPE_XML"] = envelope_xml
    globals()["_WORKER_BINNING_XML"] = binning_xml
    globals()["_WORKER_BARREL_KEYS"] = barrel_keys
    globals()["_WORKER_PARTICLE_KEYS"] = particle_keys


def _voxelize_event_index(event_idx: int) -> tuple[np.ndarray, float] | None:
    worker_array = _WORKER_ARRAY
    envelope_xml = _WORKER_ENVELOPE_XML
    binning_xml = _WORKER_BINNING_XML
    barrel_keys = _WORKER_BARREL_KEYS
    particle_keys = _WORKER_PARTICLE_KEYS

    if worker_array is None:
        raise RuntimeError("Worker array is not initialized.")
    if envelope_xml is None or binning_xml is None:
        raise RuntimeError("Worker XML paths are not initialized.")
    if barrel_keys is None or particle_keys is None:
        raise RuntimeError("Worker field keys are not initialized.")

    return _event_to_voxel_shower(
        worker_array[event_idx],
        envelope_xml=envelope_xml,
        binning_xml=binning_xml,
        barrel_keys=barrel_keys,
        particle_keys=particle_keys,
    )


def _append_to_datasets(
    h5_file: h5py.File,
    showers_batch: list[np.ndarray],
    energies_batch: list[float],
) -> tuple[h5py.Dataset, h5py.Dataset]:
    showers_np = np.stack(showers_batch, axis=0).astype(np.float32)
    energies_np = np.asarray(energies_batch, dtype=np.float32).reshape(-1, 1)

    if "showers" not in h5_file:
        num_voxels = showers_np.shape[1]
        h5_file.create_dataset(
            "showers",
            shape=(0, num_voxels),
            maxshape=(None, num_voxels),
            dtype=np.float32,
            chunks=(min(WRITE_CHUNK_SIZE, max(1, showers_np.shape[0])), num_voxels),
            compression="gzip",
        )
        h5_file.create_dataset(
            "incident_energies",
            shape=(0, 1),
            maxshape=(None, 1),
            dtype=np.float32,
            chunks=(min(WRITE_CHUNK_SIZE, max(1, energies_np.shape[0])), 1),
            compression="gzip",
        )

    showers_ds = h5_file["showers"]
    energies_ds = h5_file["incident_energies"]

    start = showers_ds.shape[0]
    end = start + showers_np.shape[0]

    showers_ds.resize((end, showers_ds.shape[1]))
    showers_ds[start:end, :] = showers_np

    energies_ds.resize((end, 1))
    energies_ds[start:end, :] = energies_np

    return showers_ds, energies_ds


def main() -> int:
    args = parse_args()

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    root_files = collect_root_files(args.input)
    if not root_files:
        raise FileNotFoundError(f"No ROOT files found at: {args.input}")

    logging.info("Found %d ROOT files to process.", len(root_files))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    expected_num_voxels: int | None = None
    total_showers = 0
    write_showers: list[np.ndarray] = []
    write_energies: list[float] = []

    with h5py.File(args.output, "w") as h5_file:
        for root_file in tqdm(root_files, desc="Processing ROOT files"):
            array = extract_calo_showers(str(root_file), tree_name=args.tree_name)
            array = preprocess_calo_showers(array)
            array = preprocess_particles(array)

            if len(array) == 0:
                continue

            barrel_keys = [key for key in array.fields if key.startswith(f"{BARREL_KEY}.")]
            particle_keys = [key for key in array.fields if key.startswith(f"{PARTICLE_KEY}.")]

            use_parallel = args.num_workers > 1 and len(array) > 1
            processed_iter = None

            if use_parallel:
                try:
                    ctx = mp.get_context("fork")
                    if ctx.get_start_method() == "fork":
                        chunksize = max(1, len(array) // (args.num_workers * 8))
                        with ctx.Pool(
                            processes=args.num_workers,
                            initializer=_init_worker,
                            initargs=(
                                array,
                                str(args.envelope_xml),
                                str(args.binning_xml),
                                barrel_keys,
                                particle_keys,
                            ),
                        ) as pool:
                            processed_iter = pool.imap_unordered(
                                _voxelize_event_index,
                                range(len(array)),
                                chunksize=chunksize,
                            )
                            for result in processed_iter:
                                if result is None:
                                    continue
                                shower, energy = result

                                if expected_num_voxels is None:
                                    expected_num_voxels = len(shower)
                                elif len(shower) != expected_num_voxels:
                                    raise ValueError(
                                        "Inconsistent voxel vector size across events: "
                                        f"expected {expected_num_voxels}, got {len(shower)} in {root_file}."
                                    )

                                write_showers.append(shower)
                                write_energies.append(energy)
                                total_showers += 1

                                if len(write_showers) >= WRITE_CHUNK_SIZE:
                                    _append_to_datasets(h5_file, write_showers, write_energies)
                                    write_showers.clear()
                                    write_energies.clear()
                        continue
                except ValueError:
                    pass

            for event_idx in range(len(array)):
                result = _event_to_voxel_shower(
                    array[event_idx],
                    envelope_xml=str(args.envelope_xml),
                    binning_xml=str(args.binning_xml),
                    barrel_keys=barrel_keys,
                    particle_keys=particle_keys,
                )
                if result is None:
                    continue

                shower, energy = result

                if expected_num_voxels is None:
                    expected_num_voxels = len(shower)
                elif len(shower) != expected_num_voxels:
                    raise ValueError(
                        "Inconsistent voxel vector size across events: "
                        f"expected {expected_num_voxels}, got {len(shower)} in {root_file}."
                    )

                write_showers.append(shower)
                write_energies.append(energy)
                total_showers += 1

                if len(write_showers) >= WRITE_CHUNK_SIZE:
                    _append_to_datasets(h5_file, write_showers, write_energies)
                    write_showers.clear()
                    write_energies.clear()

        if write_showers:
            _append_to_datasets(h5_file, write_showers, write_energies)

    if total_showers == 0:
        raise RuntimeError("No valid showers were produced from the provided ROOT input.")

    print(
        f"Saved {total_showers} showers with {expected_num_voxels} voxels each to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
