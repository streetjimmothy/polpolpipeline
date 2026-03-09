import argparse
import itertools
import os
import sys
from typing import Dict, Tuple

#!/usr/bin/env python3
"""
Compare text files (by name) between two directories and count differing lines.

Designed for very large files:
- Uses binary mode to avoid decode overhead.
- Uses large buffered readers to minimize syscalls.
- Compares line-by-line; extra lines in either file are counted as differences.
"""


DEFAULT_EXT = ".txt"
DEFAULT_BUFFER_SIZE = 8 * 1024 * 1024  # 8 MiB


def compare_files(path_a: str, path_b: str, buffer_size: int) -> Tuple[int, int]:
	"""Return (diff_count, total_lines_compared).

	diff_count: number of lines in file A that do not appear anywhere in file B.
	Normalization: both files' lines are compared after stripping leading/trailing
	whitespace and normalizing line endings (handled via `.strip()` on bytes).

	Optimized for speed with ample RAM: file B is loaded entirely into memory
	as a set of normalized lines (bytes) for O(1) average membership checks.
	"""

	def _normalize(line: bytes) -> bytes:
		# Strip leading/trailing whitespace, including CR/LF, tabs, spaces
		return line.strip()

	# Load and normalize all lines from B into a set of bytes for fast membership checks
	with open(path_b, "rb", buffering=buffer_size) as fb:
		lines_b = set(_normalize(lb) for lb in fb)

	diffs = 0
	total = 0
	# Iterate lines in A, normalize, and count those not present in B
	with open(path_a, "rb", buffering=buffer_size) as fa:
		for la in fa:
			total += 1
			nla = _normalize(la)
			if nla not in lines_b:
				diffs += 1
	return diffs, total


def collect_files(directory: str, extension: str) -> Dict[str, str]:
	"""Recursively collect files ending with the given extension.

	Returns a mapping from filename to absolute path. If duplicate filenames
	are encountered in different subdirectories, the first occurrence is kept
	and a warning is printed to stderr for subsequent duplicates.
	"""
	files: Dict[str, str] = {}
	for root, _dirs, filenames in os.walk(directory):
		for fname in filenames:
			if not fname.endswith(extension):
				continue
			fpath = os.path.join(root, fname)
			if fname in files:
				print(
					f"Warning: duplicate filename '{fname}' at {fpath}; keeping {files[fname]}",
					file=sys.stderr,
				)
				continue
			files[fname] = fpath
	return files


def main() -> int:
	parser = argparse.ArgumentParser(description="Count differing lines for matching text files across two directories.")
	parser.add_argument("--d1", help="First directory containing text files.")
	parser.add_argument("--d2", help="Second directory containing text files.")
	parser.add_argument("--ext", default=DEFAULT_EXT, help=f"File extension to match (default: {DEFAULT_EXT})")
	parser.add_argument(
		"--buffer-size",
		type=int,
		default=DEFAULT_BUFFER_SIZE,
		help=f"Buffer size in bytes for file reads (default: {DEFAULT_BUFFER_SIZE})",
	)
	args = parser.parse_args()

	if not os.path.isdir(args.d1) or not os.path.isdir(args.d2):
		print("Both arguments must be directories.", file=sys.stderr)
		return 1

	files_a = collect_files(args.d1, args.ext)
	files_b = collect_files(args.d2, args.ext)

	print(f"Found {len(files_a)} *{args.ext} files in {args.d1}")
	print(files_a['ARG_wcc-pruned_0comm_all.txt'])
	print(f"Found {len(files_b)} *{args.ext} files in {args.d2}")
	print(files_b['ARG_wcc-pruned_0comm_all.txt'])

	common_names = sorted(set(files_a) & set(files_b))
	if not common_names:
		print(f"No matching *{args.ext} files found between {args.d1} and {args.d2}.")
		return 0

	total_diffs = 0
	for name in common_names:
		path_a = files_a[name]
		path_b = files_b[name]
		diffs, total = compare_files(path_a, path_b, args.buffer_size)
		total_diffs += diffs
		print(f"{name}: {diffs} differing lines (total lines compared: {total}. ({(diffs / total * 100) if total > 0 else 0:.2f}%)")

	print(f"\nTotal differing lines across all matching files: {total_diffs}")
	return 0


if __name__ == "__main__":
	sys.exit(main())