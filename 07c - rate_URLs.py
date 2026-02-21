import csv
from typing import Dict
import utilities as utils
import os

try:
	from tqdm import tqdm  # optional progress bar
except Exception:  # pragma: no cover
	tqdm = None

domains: Dict[str, Dict[str, str]] = {}

def process_file(in_file, domain_csv, out_file, show_progress: bool = True):
	"""Process input file, matching known domains and writing labeled rows.

	Parameters
	----------
	in_file : str
		Path to text file containing tweet lines.
	domain_csv : str
		CSV of domains with label + type information.
	out_file : str
		Destination CSV to write results.
	show_progress : bool
		If True and tqdm available, display a progress bar over lines.
	"""
	with open(domain_csv, 'r', encoding='utf-8', errors='ignore') as f:
		csv_reader = csv.reader(f)
		try:
			next(csv_reader)  # skip header
		except StopIteration:
			pass
		for row in csv_reader:
			if not row:
				continue
			# Defensive: ensure row long enough
			if len(row) >= 7:
				domains[row[0]] = {"label": row[1], "type": row[6]}

	# Pre-count lines for progress total (optional)
	total_lines = None
	if show_progress and tqdm is not None:
		try:
			with open(in_file, 'rb') as counter:
				total_lines = sum(1 for _ in counter)
		except OSError:
			total_lines = None

	with open(in_file, 'r', encoding='utf-8', errors='ignore') as fin, \
		 open(out_file, 'w', encoding='utf-8', errors='ignore', newline='') as fout:
		csv_writer = csv.writer(fout)
		csv_writer.writerow(["tweet_text", "label", "type"])

		line_iter = fin
		if show_progress and tqdm is not None:
			line_iter = tqdm(fin, total=total_lines, unit='line', desc='Rating', ncols=100)

		for line in line_iter:
			# Naive substring scan; consider Aho-Corasick if domains large.
			for dom_key, dom_value in domains.items():
				if dom_key and dom_key in line:
					csv_writer.writerow([line.strip(), dom_value["label"], dom_value["type"]])
					break

if __name__ == "__main__":  # simple CLI usage
	import argparse
	parser = argparse.ArgumentParser(description="Rate URLs in a text file against a domain CSV.")
	utils.create_input_args(parser)
	utils.create_output_args(parser, suffix='-rated.csv')  # TODO: not used
	parser.add_argument("domain_csv", help="CSV containing domains and metadata")
	parser.add_argument("--verbose", action="store_true", help="Disable progress bar")
	args = parser.parse_args()

	input_files = utils.parse_input_files_arg(args.input_file, ext=".txt")

	for input_path in input_files:
		process_file(input_path, args.domain_csv, f"{os.path.splitext(input_path)[0]}-rated.csv", args.verbose)


