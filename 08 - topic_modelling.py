import os
import argparse
import utilities as util
import importlib.util
from pathlib import Path
import tqdm
import csv
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
	from cuml.feature_extraction.text import CountVectorizer
except ImportError:
	print("Could not load cuML CountVectorizer for GPU. Falling back to scikit-learn version for CPU.")
	from sklearn.feature_extraction.text import CountVectorizer

def _load_module(path: Path, module_name: str):
	"""Dynamically load a module from an arbitrary path (supports filenames with spaces)."""
	spec = importlib.util.spec_from_file_location(module_name, str(path))
	if spec is None or spec.loader is None:
		raise ImportError(f"Cannot load module {module_name} from {path}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)  # type: ignore[attr-defined]
	return module

def process_files(filename: Path, domain_csv: Path, unicode_mode: bool, verbose: bool):
	# Load modules once
	clean_mod = _load_module(Path('07a - clean_text.py'), 'clean_text_mod')
	resolve_mod = _load_module(Path('07b - resolve_URLs.py'), 'resolve_urls_mod')
	rate_mod = _load_module(Path('07c - rate_URLs.py'), 'rate_urls_mod')
	denoise_mod = _load_module(Path('07d - denoise_URLs.py'), 'denoise_urls_mod')

	# Get function refs (with basic validation)
	clean_func = getattr(clean_mod, 'process_file')
	resolve_func = getattr(resolve_mod, 'process_file')
	rate_func = getattr(rate_mod, 'process_file')
	denoise_func = getattr(denoise_mod, 'process_file')

	domain_csv = domain_csv.resolve()
	if not domain_csv.is_file():
		raise FileNotFoundError(f"Domain CSV not found: {domain_csv}")

	suffixes_to_skip = ('-cleaned.txt', '-resolved.txt', '-denoised.txt', '-rated.csv')
	if any(filename.endswith(suffix) for suffix in suffixes_to_skip):
		if verbose:
			print(f"Skipping file with processed suffix: {filename}")
		return

	file_type = 'ascii' if not unicode_mode else 'unicode'
	infile = filename
	cleaned_file = (filename[:-4] + f'_{file_type}-cleaned.txt')
	resolved_file = (filename[:-4] + f'_{file_type}-resolved.txt')
	denoised_file = (filename[:-4] + f'_{file_type}-denoised.txt')
	rated_file = (filename[:-4] + f'_{file_type}-rated.csv')

	if verbose:
		print(f"[1/3] Cleaning: {infile} -> {cleaned_file}")
	# clean_text.process_file(Path, Path, unicode_mode, verbose=...)
	clean_func(Path(infile), Path(cleaned_file), unicode_mode, verbose)

	if verbose:
		print(f"[2/3] Resolving URLs: {cleaned_file} -> {resolved_file}")
	# resolve_URLs.process_file(infile, outfile)
	resolve_func(str(cleaned_file), str(resolved_file), verbose)

	if verbose:
		print(f"[3/3] Denoising: {resolved_file} -> {denoised_file}")
	# denoise.process_file(in_file, out_file, show_progress=verbose)
	denoise_func(str(resolved_file), str(domain_csv), str(denoised_file), show_progress=verbose)

	# if verbose:
	# 	print(f"[4/3] Rating URLs: {resolved_file} -> {rated_file}")
	# # rate_URLs.process_file(in_file, domain_csv, out_file, show_progress=verbose)
	# rate_func(str(resolved_file), str(domain_csv), str(rated_file), show_progress=verbose)

	if verbose:
		print(f"Completed: {infile}")


def load_documents(path: str, document_column: str = 'tweet_text'):
	if not os.path.isfile(path):
		raise FileNotFoundError(f"Input file not found: {path}")

	documents = []
	if str(path).endswith('.csv'):
		print("Loading CSV ...")
		documents = []
		with open(path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
			reader = csv.DictReader(f)
			for row in tqdm(reader, desc=f"Extracting {document_column}", unit="row"):
				text = row.get(document_column, None)
				if text is not None:
					documents.append(text)
	else:
		with open(path, 'r', encoding='utf-8') as f:
			for line in tqdm(f, desc=f"Loading {path.name}", unit="line"):
				doc = line.strip()
				if doc:
					documents.append(doc)
	print(f"Loaded {len(documents)} rows from {path}")
	return documents


def train_vectoriser(
	docs,
	vectoriser
):
	def doc_generator():
		if not docs:
			return
		progress = tqdm(
			total=len(docs),
			desc="Vectorizing documents",
			unit="doc",
			leave=False,
		)
		for doc in docs:
			progress.update(1)
			yield doc
		progress.close()

	return vectoriser.fit_transform(doc_generator())


def load_stopwords(custom_path: Path | None):
	stopwords = set(ENGLISH_STOP_WORDS)
	if custom_path:
		if not custom_path.is_file():
			raise FileNotFoundError(f"Custom stopword file not found: {custom_path}")
		with custom_path.open("r", encoding="utf-8", errors="ignore") as f:
			for line in f:
				token = line.strip().lower()
				if token:
					stopwords.add(token)
	return sorted(stopwords)

def main():
	parser = argparse.ArgumentParser(description="Topic Modelling multirunner")
	util.create_input_args(parser)
	util.create_output_args(parser, suffix='{.csv|.png|.html}', help="Directory to save output plots and CSVs. Each stage will create its own subdirectory inside this directory")
	parser.add_argument("--stop_words", required=False, help="Custom stopwords list to use")
	parser.add_argument("--min-df", type=int, default=5, help="Minimum document frequency for CountVectorizer (default: 5)")
	parser.add_argument("--max-df", type=float, default=0.7, help="Maximum document frequency ratio (default: 0.7)")
	parser.add_argument("--verbose", action='store_true', help="Enable verbose output for debugging and progress tracking")
	#parser.add_argument("--document_column", default="tweet_text", help="Column name in the CSV that contains the document text")
	parser.add_argument("--max_topics", type=int, default=10, help="Maximum number of topics to reduce to (default: 10)")
	args = parser.parse_args()

	input_files = util.parse_input_files_arg(args.input_file, ext="-denoised.txt")
	output_files = util.parse_output_files_arg(args.output, input_files)

	stopwords = set(ENGLISH_STOP_WORDS)
	if (args.stop_words):
		print(f"Using custom stopwords from: {args.stop_words}")
		custom_stopwords = load_stopwords(Path(args.stop_words))
		print(f"Loaded {len(custom_stopwords)} custom stopwords")
		
	scripts = {
		"08a - BERTopic.py" : {"min_cluster_size": 150},
		# "08b - LDA.py" : {},
		# "08c - LDA_then_BERTopic.py" : {},
		# "08d - LDA_powered_BERTopic.py" : {},
		# "08e - BERTopic_then_LDA.py": {},
		# "08f - TF-IDF.py": {}
	}

	
	for input_file, output_file in zip(input_files, output_files):
		print(f"Processing input file: {input_file}")
		documents = load_documents(input_file)
		print(f"Generating term-frequency matrices with CountVectorizer ...")
		term_doc_matrix = train_vectoriser(
			documents, 
			CountVectorizer(
				stop_words=stopwords,
				lowercase=True,
				min_df=args.min_df,
				max_df=args.max_df,
				binary=False,
				ngram_range=(1, 2),
			)
		)
		for script, args in scripts.items():
			module = _load_module(Path(script), script.split(" ")[:-1].replace(".py", "_mod"))
			func = getattr(module, 'run')
			func_args = args + {
				"input_file": input_file,
				"output_file_base": output_file,
				"verbose": args.verbose,
				"vectorised_documents": term_doc_matrix,
				"max_topics": args.max_topics
			}
			print(f"Running {script} with args: {func_args}")
			func(*func_args)


if __name__ == "__main__":
	main()