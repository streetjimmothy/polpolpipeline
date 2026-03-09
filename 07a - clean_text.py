#!/usr/bin/env python3
import unicodedata
from pathlib import Path
import sys
import argparse
import re
import os
from typing import Optional
import utilities as utils

try:
	# tqdm is optional; if missing we silently skip progress bars
	from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
	tqdm = None  # fallback sentinel

ALLOWED_WHITESPACE = {"\n", "\r", "\t"}

# Precompiled regex for fast ASCII cleaning: keep 0x20–0x7E plus allowed whitespace.
_ASCII_ALLOWED_BYTES = b"\n\r\t" + bytes(range(0x20, 0x7F))
# Negated class of allowed chars (everything except space–tilde and allowed whitespace)
ASCII_DISALLOWED_RE = re.compile(rb"[^\x20-\x7E\n\r\t]+")


def ascii_fast_clean_bytes(data: bytes) -> bytes:
	# Remove disallowed bytes in one pass (works on bytes, fastest).
	return ASCII_DISALLOWED_RE.sub(b"", data)


def build_unicode_translate_table():
	# Build once: map disallowed ordinals to None for str.translate
	table = {}
	for cp in range(0x110000):
		ch = chr(cp)
		cat = unicodedata.category(ch)
		if cat.startswith("C") and ch not in ALLOWED_WHITESPACE:
			table[cp] = None
		else:
			# Drop unprintable (isprintable False)
			if not ch.isprintable() and ch not in ALLOWED_WHITESPACE:
				table[cp] = None
	return table


_UNICODE_TRANSLATE_TABLE = None  # lazy init


def unicode_clean(text: str) -> str:
	global _UNICODE_TRANSLATE_TABLE
	if _UNICODE_TRANSLATE_TABLE is None:
		_UNICODE_TRANSLATE_TABLE = build_unicode_translate_table()
	return text.translate(_UNICODE_TRANSLATE_TABLE)


def process_file(input_path: Path, output_path: Path, unicode_mode: bool, verbose: bool = False, chunk_size: int = 4 * 1024 * 1024):
	"""Clean a single file.

	Parameters
	----------
	input_path : Path
		Source file path.
	output_path : Path
		Destination (will be overwritten).
	unicode_mode : bool
		If True, perform Unicode printable filtering; else fast ASCII whitelist.
	verbose : bool
		If True, show a tqdm progress bar and summary.
	chunk_size : int
		Read size in bytes per iteration (default 4 MiB). Large enough for throughput, small enough for progress updates.
	"""

	total_size: Optional[int] = None
	if verbose:
		try:
			total_size = input_path.stat().st_size
		except OSError:
			total_size = None

	progress = None
	if verbose and tqdm is not None:
		progress = tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Cleaning {input_path.name}", leave=False)

	if not unicode_mode:
		# Stream bytes, clean, and write directly
		with input_path.open("rb") as fin, output_path.open("wb") as fout:
			for chunk in iter(lambda: fin.read(chunk_size), b""):
				cleaned = ascii_fast_clean_bytes(chunk)
				fout.write(cleaned)
				if progress is not None:
					progress.update(len(chunk))
	else:
		# Read in chunks, decode incrementally, accumulate, then clean via translate (same memory as original approach)
		text_parts: list[str] = []
		read_bytes = 0
		with input_path.open("rb") as fin:
			for chunk in iter(lambda: fin.read(chunk_size), b""):
				read_bytes += len(chunk)
				# Decode each chunk ignoring undecodable sequences at boundaries
				text_parts.append(chunk.decode("utf-8", errors="ignore"))
				if progress is not None:
					progress.update(len(chunk))
		combined = ''.join(text_parts)
		cleaned_text = unicode_clean(combined)
		with output_path.open("wb") as fout:
			fout.write(cleaned_text.encode("utf-8"))

	if progress is not None:
		progress.close()
		if verbose:
			try:
				out_size = output_path.stat().st_size
				ratio = (out_size / total_size) if total_size and total_size > 0 else 0
				print(f"Cleaned {input_path.name}: {total_size} -> {out_size} bytes ({ratio:.2%} retained)")
			except OSError:
				pass


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Remove non-decodable and non-printable characters. Default: fast ASCII whitelist. Cleaned files saved to \"-cleaned.txt\" files."
	)
	utils.create_input_args(parser)
	utils.create_output_args(parser, suffix='_{unicode|ascii}-cleaned.txt') #TODO: not used
	parser.add_argument("--verbose", action='store_true', help="Enable verbose output for debugging and progress tracking")
	parser.add_argument("--unicode", action="store_true", help="Use Unicode printable filtering (slower).")

	args = parser.parse_args()
	input_files=utils.parse_input_files_arg(args.input_file, ext=".txt")

	file_type = 'ascii' if not args.unicode else 'unicode'
	try:
		for input_path in input_files:
			process_file(Path(input_path), Path(f"{os.path.splitext(input_path)[0]}_{file_type}-cleaned.txt"), args.unicode, verbose=args.verbose)
	except Exception as e:
		print(f"Processing failed: {e}", file=sys.stderr)
		sys.exit(2)
