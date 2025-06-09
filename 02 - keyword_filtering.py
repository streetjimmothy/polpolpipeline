import re
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import os
import itertools
import argparse

import time
import humanize
import psutil

def load_keywords(keywords_file):
	"""
	Load keywords from a file or use a predefined list
	"""
	with open(keywords_file, 'r', encoding='utf-8') as f:

		# Read keywords one per line and strip whitespace
		keywords = []
		for line in f:
			line=line.strip()
			keywords.append(line)
	print(f"Loaded {len(keywords)} keywords from {keywords_file}")
	print(f"Using {len(keywords)} predefined keywords")
	
	return keywords

keywords_pattern = None

def check_tweet_bytes(tweet_bytes):
	if keywords_pattern.search(tweet_bytes):
		return tweet_bytes
	return None

def search_tweets_raw(input_file, output_file, keywords_file, chunk_size=10000000, mp_chunksize=50000):
	total_processed = 0
	total_matches = 0
	
	keywords = load_keywords(keywords_file)
	# Compile the regex pattern for bytes
	global keywords_pattern
	bytes_keywords = [kw.lower().encode('utf-8') for kw in keywords]
	pattern_string = b'|'.join(b'\\b' + re.escape(kw) + b'\\b' for kw in bytes_keywords)
	keywords_pattern = re.compile(pattern_string, re.IGNORECASE)
	
	with open(output_file, 'wb') as out_file:
		with mp.Pool(processes=mp.cpu_count()) as pool:
			with open(input_file, 'rb') as in_file:
				pbar = tqdm(desc="Processing file", unit="lines")
				
				while True:
					# Read a chunk of lines
					chunk = list(itertools.islice(in_file, chunk_size))
					if not chunk:
						break
					
					chunk_len = len(chunk)
					total_processed += chunk_len
					
					# Process the chunk with multiprocessing
					chunk_results = pool.imap(check_tweet_bytes, chunk, chunksize=mp_chunksize)
					
					# Track progress with tqdm for this chunk
					chunk_matches = 0
					for result in tqdm(chunk_results, total=chunk_len, desc=f"Chunk {total_processed//chunk_size}", leave=False):
						if result is not None:
							out_file.write(result)
							chunk_matches += 1
					
					total_matches += chunk_matches
					pbar.update(chunk_len)
					pbar.set_postfix(matches=total_matches)
				
				pbar.close()
	
	print(f"Completed: {total_processed} lines processed, {total_matches} matches found ({total_matches/total_processed:.2%})")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Filter tweets by keywords")
	parser.add_argument("-i", "--input_file", required=True, help="Path to the input file containing tweets")
	parser.add_argument("-k", "--keywords_file", required=True, help="Path to the file containing keywords")
	parser.add_argument("-o", "--output_file", required=False, help="Path to save the filtered tweets (defaults to the country name in the keyword filename)")
	parser.add_argument("--chunk_size", type=int, default=100000000, help="Number of bytes to read at a time (default: 100MB)")
	parser.add_argument("--mp_chunksize", type=int, default=50000, help="Chunk size for multiprocessing (default: 50000)")
	
	args = parser.parse_args()
	
	start_time = time.time()
	
	if not os.path.exists(args.input_file):
		print(f"Input file does not exist: {args.input_file}")
		exit(1)
	if not os.path.exists(args.keywords_file):
		print(f"Keywords file does not exist: {args.keywords_file}")
		exit(1)
	if not args.output_file:
		# Use the country name from the keywords file as the output filename
		country_name = os.path.splitext(os.path.basename(args.keywords_file))[0][-3:]	#the country name is the last 3 characters of the keywords file name
		input_file = os.path.splitext(os.path.basename(args.input_file))[0]
		if not os.path.exists(country_name):
			os.makedirs(country_name)
		args.output_file = f"{country_name}/{country_name}_{input_file}.txt"
		print(f"Output file not specified, using default: {args.output_file}")
	if os.path.exists(args.output_file):
		input(f"Output file exists: {args.output_file}; Confirm overwrite (Ctrl+C to cancel)?")
	search_tweets_raw(args.input_file, args.output_file, args.keywords_file, args.chunk_size, args.mp_chunksize)
	
	elapsed_time = time.time() - start_time
	print(f"Total time taken: {humanize.naturaldelta(elapsed_time)}")
	print(f"Memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")