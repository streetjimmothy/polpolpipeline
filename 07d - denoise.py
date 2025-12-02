import argparse
import sys, os
from pathlib import Path
import re
from tqdm import tqdm

### TODO Doesn't do enough to respect quotation marks and newlines, messes up csvs
end_of_url = rb"[\"|,|\n|^|\s]"
web_common = rb"http://|https://|www\.|\.html|\.htm|\.php|\.asp|\.aspx|\.jsp|\.cfm|\.cgi|\.pl|\.xml"
twitter_common = rb"x.com/|twitter.com/|mobile.twitter.com/|m.twitter.com/|\d+/video/\d|\d+/photo/\d|/status/"
numeric_sequences_not_dates = rb"\d{5,}"
header_links = rb"#\W.+?"+end_of_url
dates = rb"/\d{2,4}"
cnn_header_links = rb"(#h.+?)(?="+end_of_url+rb")"
trailing_ids = rb"[-_]\d+?(?="+end_of_url+rb")|_n_.+?(?="+end_of_url+rb")"
UUID = rb"[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}"
ASCII_DISALLOWED_RE = re.compile(
	web_common + b"|" + 
	twitter_common + b"|" + 
	numeric_sequences_not_dates + b"|" + 
	header_links + b"|" + 
	trailing_ids + b"|" + 
	dates + b"|" + 
	cnn_header_links + b"|" + 
	UUID, 
	re.IGNORECASE
)


query_strings = rb".*(\?.*?=.*?(?="+end_of_url+rb"))"
query_strings_pattern = re.compile(query_strings)
def replace_group(match):
	# Replace the second group with an empty string
	full_match = match.group(0)
	to_replace = match.group(1)
	terminator = to_replace[-1:] if to_replace else b""
	return full_match.replace(to_replace, terminator)

def ascii_fast_clean_bytes(data: bytes) -> bytes:
	data = ASCII_DISALLOWED_RE.sub(b"", data)
	data = query_strings_pattern.sub(replace_group, data)
	return data

def process_file(input_path: Path, output_path: Path, verbose: bool = False, chunk_size: int = 6 * 1024 * 1024):
	"""Clean a single file.

	Parameters
	----------
	input_path : Path
		Source file path.
	output_path : Path
		Destination file path.
	unicode_mode : bool
		If True, use Unicode cleaning; otherwise, use fast ASCII cleaning.
	verbose : bool
		If True, print progress information.
	chunk_size : int
		Size of chunks to read and process at a time (in bytes).
	"""
	if verbose:
		print(f"Processing file: {input_path} -> {output_path}")
	total_bytes = os.path.getsize(input_path)
	processed_bytes = 0

	with open(input_path, 'rb') as infile, open(output_path, 'wb') as outfile:
		if verbose:
			pbar = tqdm(total=total_bytes, unit='B', unit_scale=True, desc=f"Cleaning {input_path.name}", ncols=100)
		else:
			pbar = None
		while True:
			data = infile.read(chunk_size)
			if not data:
				break
			cleaned_data = ascii_fast_clean_bytes(data)
			outfile.write(cleaned_data)
			processed_bytes += len(data)
			if pbar:
				pbar.update(len(data))
		if pbar:
			pbar.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Remove non-decodable and non-printable characters. Default: fast ASCII whitelist. Cleaned files saved to \"-cleaned.txt\" files."
	)
	parser.add_argument("-i", "--input_file", "--input_files", "--input_dir", nargs='+', required=True, help="Path to the input file containing text (multiple files can be specified, they will each be cleaned. A directory can also be specified, in which case all text files in that directory will be cleaned)")
	parser.add_argument("--verbose", action='store_true', help="Enable verbose output for debugging and progress tracking")

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
		for input_path in args.input_file:
			process_file(Path(input_path), Path(f"{os.path.splitext(input_path)[0]}-denoised.txt"), verbose=args.verbose)
	except Exception as e:
		print(f"Processing failed: {e}", file=sys.stderr)
		sys.exit(2)
