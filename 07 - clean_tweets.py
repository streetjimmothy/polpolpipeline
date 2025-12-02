import os
import sys
import argparse
import importlib.util
from pathlib import Path


def _load_module(path: Path, module_name: str):
	"""Dynamically load a module from an arbitrary path (supports filenames with spaces)."""
	spec = importlib.util.spec_from_file_location(module_name, str(path))
	if spec is None or spec.loader is None:
		raise ImportError(f"Cannot load module {module_name} from {path}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)  # type: ignore[attr-defined]
	return module


def process_files(directory: Path, domain_csv: Path, unicode_mode: bool, verbose: bool):
	# Load modules once
	clean_mod = _load_module(Path('07a - clean_text.py'), 'clean_text_mod')
	resolve_mod = _load_module(Path('07b - resolve_URLs.py'), 'resolve_urls_mod')
	rate_mod = _load_module(Path('07c - rate_URLs.py'), 'rate_urls_mod')

	# Get function refs (with basic validation)
	clean_func = getattr(clean_mod, 'process_file')
	resolve_func = getattr(resolve_mod, 'process_file')
	rate_func = getattr(rate_mod, 'process_file')

	domain_csv = domain_csv.resolve()
	if not domain_csv.is_file():
		raise FileNotFoundError(f"Domain CSV not found: {domain_csv}")

	for filename in os.listdir(directory):
		if not filename.endswith('.txt'):
			continue
		if any(filename.endswith(suffix) for suffix in ('-cleaned.txt', '-resolved.txt', '-done.txt')):
			continue

		infile = directory / filename
		cleaned_file = directory / (filename[:-4] + '-cleaned.txt')
		resolved_file = directory / (filename[:-4] + '-resolved.txt')
		done_file = directory / (filename[:-4] + '-done.txt')

		if verbose:
			print(f"[1/3] Cleaning: {infile} -> {cleaned_file}")
		# clean_text.process_file(Path, Path, unicode_mode, verbose=...)
		clean_func(Path(infile), Path(cleaned_file), unicode_mode, verbose)

		if verbose:
			print(f"[2/3] Resolving URLs: {cleaned_file} -> {resolved_file}")
		# resolve_URLs.process_file(infile, outfile)
		resolve_func(str(cleaned_file), str(resolved_file))

		if verbose:
			print(f"[3/3] Rating URLs: {resolved_file} -> {done_file}")
		# rate_URLs.process_file(in_file, domain_csv, out_file, show_progress=verbose)
		rate_func(str(resolved_file), str(domain_csv), str(done_file), show_progress=verbose)

		if verbose:
			print(f"Completed: {done_file}\n")


def main(argv=None):
	parser = argparse.ArgumentParser(description='Pipeline: clean, resolve, rate tweet text files in a directory.')
	parser.add_argument('directory', help='Directory containing raw .txt files.')
	parser.add_argument('--domain-csv', default='07c - domain_list_clean.csv', help='Domain CSV (default: 07c - domain_list_clean.csv)')
	parser.add_argument('--unicode', action='store_true', help='Use Unicode cleaning mode (slower).')
	parser.add_argument('--verbose', action='store_true', help='Verbose logging & progress bars if available.')
	args = parser.parse_args(argv)

	directory = Path(args.directory)
	if not directory.is_dir():
		print(f"Directory not found: {directory}", file=sys.stderr)
		sys.exit(1)

	try:
		process_files(directory, Path(args.domain_csv), args.unicode, args.verbose)
	except Exception as e:
		print(f"Pipeline failed: {e}", file=sys.stderr)
		sys.exit(2)


if __name__ == '__main__':
	main()