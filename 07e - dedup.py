import argparse
import utilities as utils
from pathlib import Path
import os
from tqdm import tqdm

def process_file(input_path: Path, output_path: Path, verbose: bool = False) -> None:
	with open(input_path, 'r', encoding='utf-8') as f:
		if verbose:
			lines = [line for line in tqdm(f, desc=f"Reading {input_path.name}", unit="lines")]
		else:
			lines = f.readlines()
	if verbose:
		print(f"Total lines: {len(lines)}")
	deduped_lines = utils.deduplicate_list(lines)
	if verbose:
		print(f"Total unique lines: {len(deduped_lines)}")
		print(f"% unique: {len(deduped_lines) / len(lines) * 100:.2f}%")
	with open(output_path, 'w', encoding='utf-8') as f:
		if verbose:
			for line in tqdm(deduped_lines, desc=f"Writing {output_path.name}", unit="lines"):
				f.write(line)
		else:
			f.writelines(deduped_lines)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Deduplicate lines in a text file or csv.")
	utils.create_input_args(parser, ext="-denoised.[csv|txt]")	#TODO: proper csv support 
	utils.create_output_args(parser, suffix='-deduped.[csv|txt]')  # TODO: not used
	parser.add_argument("--verbose", action='store_true', help="Enable verbose output for debugging and progress tracking")

	args = parser.parse_args()
	input_files = utils.parse_input_files_arg(args.input_file, ext="-denoised.txt")
	for input_path in input_files:
		process_file(Path(input_path), Path(f"{os.path.splitext(input_path)[0]}-deduped.txt"), verbose=args.verbose)
