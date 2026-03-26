#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Find all HDF5 files under a folder and combine matching datasets "
			"into one output HDF5 file."
		)
	)
	parser.add_argument("input_dir", type=Path, help="Root folder to scan recursively")
	parser.add_argument("output_file", type=Path, help="Destination HDF5 file")
	return parser.parse_args()


def find_hdf5_files(root: Path, output_file: Path) -> list[Path]:
	patterns = ("*.h5", "*.hdf5")
	files: list[Path] = []
	output_resolved = output_file.resolve()
	for pattern in patterns:
		for path in root.rglob(pattern):
			if not path.is_file():
				continue
			if path.resolve() == output_resolved:
				continue
			files.append(path)
	return sorted(set(files))


def list_dataset_paths(h5_file: h5py.File) -> set[str]:
	paths: set[str] = set()

	def visitor(name: str, obj: h5py.Dataset) -> None:
		if isinstance(obj, h5py.Dataset):
			paths.add(name)

	h5_file.visititems(visitor)
	return paths


def validate_dataset_layout(reference: h5py.File, current: h5py.File, file_path: Path) -> None:
	ref_paths = list_dataset_paths(reference)
	cur_paths = list_dataset_paths(current)

	if ref_paths != cur_paths:
		missing = sorted(ref_paths - cur_paths)
		extra = sorted(cur_paths - ref_paths)
		raise ValueError(
			f"Dataset mismatch in {file_path}. Missing: {missing}, extra: {extra}"
		)

	for dset_path in sorted(ref_paths):
		ref_ds = reference[dset_path]
		cur_ds = current[dset_path]

		if ref_ds.dtype != cur_ds.dtype:
			raise ValueError(
				f"Dtype mismatch for dataset '{dset_path}' in {file_path}: "
				f"expected {ref_ds.dtype}, got {cur_ds.dtype}"
			)

		if ref_ds.ndim != cur_ds.ndim:
			raise ValueError(
				f"Rank mismatch for dataset '{dset_path}' in {file_path}: "
				f"expected {ref_ds.ndim}D, got {cur_ds.ndim}D"
			)

		if ref_ds.ndim == 0:
			continue

		if ref_ds.shape[1:] != cur_ds.shape[1:]:
			raise ValueError(
				f"Shape mismatch for dataset '{dset_path}' in {file_path}: "
				f"expected trailing shape {ref_ds.shape[1:]}, got {cur_ds.shape[1:]}"
			)


def ensure_parent_groups(h5_file: h5py.File, dset_path: str) -> None:
	parent = dset_path.rsplit("/", 1)[0] if "/" in dset_path else ""
	if parent:
		h5_file.require_group(parent)


def initialize_output_layout(output_h5: h5py.File, reference_h5: h5py.File) -> None:
	if list_dataset_paths(output_h5):
		return

	for dset_path in sorted(list_dataset_paths(reference_h5)):
		src = reference_h5[dset_path]
		ensure_parent_groups(output_h5, dset_path)

		if src.ndim == 0:
			output_h5.create_dataset(dset_path, data=src[()], dtype=src.dtype)
			continue

		empty_shape = (0,) + src.shape[1:]
		max_shape = (None,) + src.shape[1:]
		output_h5.create_dataset(
			dset_path,
			shape=empty_shape,
			maxshape=max_shape,
			dtype=src.dtype,
			chunks=src.chunks if src.chunks is not None else True,
			compression=src.compression,
			compression_opts=src.compression_opts,
			shuffle=src.shuffle,
			fletcher32=src.fletcher32,
		)


def append_file_to_output(output_h5: h5py.File, input_h5: h5py.File, input_path: Path) -> int:
	appended_rows = 0
	for dset_path in sorted(list_dataset_paths(input_h5)):
		src = input_h5[dset_path]
		dst = output_h5[dset_path]

		if src.ndim == 0:
			if dst[()] != src[()]:
				raise ValueError(
					f"Scalar dataset '{dset_path}' differs in {input_path}. "
					"Scalar datasets must be identical across all files."
				)
			continue

		n_rows = src.shape[0]
		if n_rows == 0:
			continue

		old_rows = dst.shape[0]
		dst.resize(old_rows + n_rows, axis=0)
		dst[old_rows : old_rows + n_rows] = src[...]

		if appended_rows == 0:
			appended_rows = n_rows
		elif appended_rows != n_rows:
			raise ValueError(
				f"Inconsistent first-axis size in {input_path}. "
				"All non-scalar datasets in a file must have the same row count."
			)

	return appended_rows


def main() -> int:
	args = parse_args()

	input_dir = args.input_dir.resolve()
	output_file = args.output_file.resolve()

	if not input_dir.exists() or not input_dir.is_dir():
		raise FileNotFoundError(f"Input directory not found: {input_dir}")
	if output_file.exists():
		os.remove(output_file)

	output_file.parent.mkdir(parents=True, exist_ok=True)

	input_files = find_hdf5_files(input_dir, output_file)
	if not input_files:
		raise FileNotFoundError(f"No HDF5 files found under: {input_dir}")

	print(f"Found {len(input_files)} file(s) to combine")
	print(f"Output: {output_file}")

	with h5py.File(input_files[0], "r") as reference_h5:
		with h5py.File(output_file, "a") as output_h5:
			initialize_output_layout(output_h5, reference_h5)

			# Validate output schema if the output file already existed.
			validate_dataset_layout(reference_h5, output_h5, output_file)

			total_rows = 0
			for index, input_path in enumerate(input_files, start=1):
				with h5py.File(input_path, "r") as input_h5:
					validate_dataset_layout(reference_h5, input_h5, input_path)
					rows = append_file_to_output(output_h5, input_h5, input_path)
					total_rows += rows

				print(
					f"[{index}/{len(input_files)}] Appended {rows} row(s) from {input_path}"
				)

			print(f"Done. Total appended rows: {total_rows}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
