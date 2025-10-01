#!/usr/bin/env python3
import unicodedata
from pathlib import Path
import sys
import argparse
import re
import os

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


def process_file(input_path: Path, output_path: Path, unicode_mode: bool, chunk_size: int = 1024 * 1024 * 1024):
	# Streaming to keep memory low
	with input_path.open("rb") as fin, output_path.open("wb") as fout:
		if not unicode_mode:
			# Fast ASCII path on raw bytes
			for chunk in iter(lambda: fin.read(chunk_size), b""):
				fout.write(ascii_fast_clean_bytes(chunk))
		else:
			# Unicode path: need to decode (ignore undecodable)
			# Read whole file if size reasonable; else chunked decode accumulate.
			raw = fin.read()
			text = raw.decode("utf-8", errors="ignore")
			cleaned = unicode_clean(text)
			fout.write(cleaned.encode("utf-8"))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Remove non-decodable and non-printable characters. Default: fast ASCII whitelist. Cleaned files saved to \"-cleaned.txt\" files."
	)
	parser.add_argument("-i", "--input_file", "--input_files", "--input_dir", nargs='+', required=True, help="Path to the input file containing text (multiple files can be specified, they will each be cleaned. A directory can also be specified, in which case all text files in that directory will be cleaned)")
	parser.add_argument("--verbose", action='store_true', help="Enable verbose output for debugging and progress tracking")
	parser.add_argument("--unicode", action="store_true", help="Use Unicode printable filtering (slower).")

	args = parser.parse_args()
	if not args.input_file:
		print("No input file specified")
		sys.exit(1)
	if len(args.input_file) > 1:
		print(f"Multiple input files specified: {args.input_file}")
	else:
		if os.path.isdir(args.input_file[0]):
			# If a directory is specified, get all text files in that directory
			args.input_file = [os.path.join(args.input_file[0], f) for f in os.listdir(
				args.input_file[0]) if f.endswith('.txt')]
			if not args.input_file:
				print(f"No text files found")
				sys.exit(1)
			print(f"Input directory specified, using files: {args.input_file}")
		else:
			print(f"Single input file specified: {args.input_file[0]}")
	try:
		for input_path in enumerate(args.input_file):
			process_file(Path(input_path), Path(f"{os.path.splitext(input_path)[0]}-cleaned.txt"), args.unicode)
	except Exception as e:
		print(f"Processing failed: {e}", file=sys.stderr)
		sys.exit(2)
