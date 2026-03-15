from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.root_utils import (  # noqa: E402
    BARREL_KEY,
    PARTICLE_KEY,
    compute_relative_position,
    extract_calo_showers,
    preprocess_calo_showers,
    preprocess_particles,
)
from scripts.voxelize import digitize_shower, get_voxels  # noqa: E402

_WORKER_ARRAY: Any = None
_WORKER_ENVELOPE_XML: str | None = None
_WORKER_BINNING_XML: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ROOT calorimeter showers to voxelized HDF5 datasets."
        )
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
    # parser.add_argument(
    #     "--incident-energy",
    #     required=True,
    #     type=float,
    #     help="Incident energy value assigned to every shower.",
    # )
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


def event_to_voxel_shower(
    event_data,
    envelope_xml: str,
    binning_xml: str,
) -> np.ndarray | None:
    barrel_keys = [
        key for key in event_data.fields if key.startswith(f"{BARREL_KEY}.")
    ]
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
    particle_df = particle_df[particle_df["MCParticles_generatorStatus"] == 1].reset_index(drop=True)
    assert len(particle_df) == 1, "Expected exactly one primary particle per event."
    momentum = particle_df[["MCParticles_momentum_x", "MCParticles_momentum_y", "MCParticles_momentum_z"]].to_numpy()
    momentum = np.linalg.norm(momentum, axis=1) * 1000 # convert to MeV

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


def _init_worker(envelope_xml: str, binning_xml: str) -> None:
    globals()["_WORKER_ENVELOPE_XML"] = envelope_xml
    globals()["_WORKER_BINNING_XML"] = binning_xml


def _voxelize_event_index(event_idx: int) -> np.ndarray | None:
    worker_array = _WORKER_ARRAY
    envelope_xml = _WORKER_ENVELOPE_XML
    binning_xml = _WORKER_BINNING_XML

    if worker_array is None:
        raise RuntimeError("Worker array is not initialized.")
    if envelope_xml is None or binning_xml is None:
        raise RuntimeError("Worker XML paths are not initialized.")

    # pylint: disable=unsubscriptable-object
    event_data = worker_array[event_idx]
    return event_to_voxel_shower(
        event_data,
        envelope_xml=str(envelope_xml),
        binning_xml=str(binning_xml),
    )


def main() -> int:
    args = parse_args()

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    root_files = collect_root_files(args.input)
    if not root_files:
        raise FileNotFoundError(f"No ROOT files found at: {args.input}")

    showers: list[np.ndarray] = []
    incident_energies = []
    expected_num_voxels: int | None = None

    for root_file in root_files:
        array = extract_calo_showers(str(root_file), tree_name=args.tree_name)
        array = preprocess_calo_showers(array)
        array = preprocess_particles(array)

        if len(array) == 0:
            continue

        use_parallel = args.num_workers > 1 and len(array) > 1

        if use_parallel:
            globals()["_WORKER_ARRAY"] = array

            try:
                ctx = mp.get_context("fork")
            except ValueError:
                ctx = mp.get_context()

            if ctx.get_start_method() != "fork":
                use_parallel = False

            if use_parallel:
                chunksize = max(1, len(array) // (args.num_workers * 4))
                with ctx.Pool(
                    processes=args.num_workers,
                    initializer=_init_worker,
                    initargs=(str(args.envelope_xml), str(args.binning_xml)),
                ) as pool:
                    shower_iter = pool.imap(
                        _voxelize_event_index,
                        range(len(array)),
                        chunksize=chunksize,
                    )
                    for shower in shower_iter:
                        if shower is None:
                            continue

                        if expected_num_voxels is None:
                            expected_num_voxels = len(shower)
                        elif len(shower) != expected_num_voxels:
                            raise ValueError(
                                (
                                    "Inconsistent voxel vector size across "
                                    "events: "
                                )
                                + (
                                    f"expected {expected_num_voxels}, "
                                    f"got {len(shower)} in {root_file}."
                                )
                            )

                        showers.append(shower[0])
                        incident_energies.append(shower[1])

        if not use_parallel:
            for event_idx in range(len(array)):
                shower = event_to_voxel_shower(
                    array[event_idx],
                    envelope_xml=str(args.envelope_xml),
                    binning_xml=str(args.binning_xml),
                )
                if shower is None:
                    continue

                if expected_num_voxels is None:
                    expected_num_voxels = len(shower)
                elif len(shower) != expected_num_voxels:
                    raise ValueError(
                        "Inconsistent voxel vector size across events: "
                        + (
                            f"expected {expected_num_voxels}, "
                            f"got {len(shower)} in {root_file}."
                        )
                    )

                showers.append(shower[0])
                incident_energies.append(shower[1])

    if not showers:
        raise RuntimeError(
            "No valid showers were produced from the provided ROOT input."
        )

    showers_array = np.stack(showers, axis=0).astype(np.float64)
    incident_energies = np.array(incident_energies).reshape(-1, 1).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as h5_file:
        h5_file.create_dataset(
            "showers",
            data=showers_array,
            compression="gzip",
        )
        h5_file.create_dataset(
            "incident_energies",
            data=incident_energies,
            compression="gzip",
        )

    print(
        (
            f"Saved {showers_array.shape[0]} showers with "
            f"{showers_array.shape[1]} voxels each to {args.output}"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
