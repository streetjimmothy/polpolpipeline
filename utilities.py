import argparse
from pathlib import Path
import os

def create_input_args(parser: argparse.ArgumentParser, ext:str = "txt", help: str | None = None) -> None:
	parser.add_argument(
		"-i",
		"--input_file",
		"--input_files",
		"--input_dir",
		nargs='+',
		required=True,
		help=help if help is not None else f"Path to the input {ext} file (multiple files can be specified, they will be process in turn. A directory can also be specified, in which case all {ext} files in that directory will be used)"
	)


def create_output_args(parser: argparse.ArgumentParser, suffix: str = "", help: str | None = None, ) -> None:
	parser.add_argument(
		"-o",
		"--output",
		required=False,
		help=help if help is not None else f"File or folder to save outputs. One input file can have a single output file. If there are multiple input files, this must be a directory - the output files will be generated based on the input file and the suffix {suffix}. If not provided, defaults to the input directory."
	)


def parse_input_files_arg(input_files: list[str] | str, ext: str | None =None) -> list[str]:
	if isinstance(input_files, str):
		input_files = [input_files]
	
	input_files = [os.path.expanduser(f) if '~' in f else f for f in input_files]
	
	if len(input_files) > 1:
		print(f"Multiple input files specified: {input_files}")
		return input_files
	else:
		if os.path.isdir(input_files[0]):
			# If a directory is specified, get all JSON files in that directory
			_input_files = []
			for f in os.listdir(input_files[0]):
				if ext is None or f.lower().endswith(ext.lower()):
					_input_files.append(os.path.join(input_files[0], f))
			if not _input_files:
				if ext:
					print(f"No {ext} files found in directory: {input_files[0]}")
				else:
					print(f"No files found in directory: {input_files[0]}")
				return []
			print(f"Input directory specified, using files: {_input_files}")
			return _input_files
		else:
			print(f"Single input file specified: {input_files[0]}")
			return [input_files[0]]


def parse_output_files_arg(output: str | None, input_files: list[str]) -> list[str] | str:
	if output is None:
		output_files = []
		for input_file in input_files:
			output_files.append(os.path.splitext(input_file)[0])
		return output_files
	
	if '~' in output:
		output = os.path.expanduser(output)

	if os.path.isdir(output):
		print(f"Output directory specified: {output}")
		output_files = []
		for input_file in input_files:
			input_filename = os.path.basename(input_file)
			output_files.append(os.path.join(output, os.path.splitext(input_filename)[0]))
		print(f"Output files: {output_files}")
		return output_files
	if len(input_files) > 1:
		output_path = Path(output)
		output_path.mkdir(parents=True, exist_ok=True)
		if not os.path.isdir(output):
			raise ValueError("When multiple input files are specified, the output must be a directory")

	return output	#a single input file and a single output file - the lack of list indicates the single file case

def get_community_label(community: str | int, community_info_path: str | None) -> str:
	if community_info_path is None:
		return str(community)
	else:
		import json
		with open(community_info_path, 'r') as f:
			community_info = json.load(f)
			community_labels = community_info.get("labels", "")
			if isinstance(community, int):
				community = str(community)
			if community in community_labels:
				community = community_labels[community]
			else:
				community = str(community)
			
			return community


def get_community_colour(community: str | int, community_info_path: str | None) -> str:
	if community_info_path is None:
		# Default colours
		default_colours = [
			"blue", "orange", "green", "red", "purple",
			"brown", "pink", "gold", "olive", "cyan"
		]
		try:
			index = int(community) % len(default_colours)
			return default_colours[index]
		except ValueError:
			return "grey"  # Fallback colour for non-integer community labels
	else:
		import json
		with open(community_info_path, 'r') as f:
			community_info = json.load(f)
			community_colours = community_info.get("colours", {})
			community = get_community_label(community, community_info_path)
			
			return community_colours.get(community, "grey")


	# Load community colours
	import json
	with open(args.community_colours, 'r') as f:
		community_info = json.load(f)
		community_colours = community_info.get("colours", "")
		community_labels = community_info.get("labels", "")


def deduplicate_list(input_list: list[str], verbose: bool = False) -> list[str]:
	deduplicated = set()
	
	if verbose:
		print(f"Deduplicating list of length {len(input_list)}")
		import tqdm
		iterator = tqdm.tqdm(input_list, desc="Deduplicating", unit="items")
	else:
		iterator = input_list
	for item in iterator:
		deduplicated.add(item)
	return list(deduplicated)