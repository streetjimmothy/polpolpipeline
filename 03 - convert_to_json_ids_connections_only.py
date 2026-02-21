import re
import json
import os
import argparse

import time
import humanize
import psutil

def extract_tweet_network_data(input_file_path, output_file_path):
	"""
	Extract user_id, tweet_id, connected_user, connected_tweet, and connection_type from tweet data
	and save to a new JSON file
	"""
	print(f"Processing {input_file_path}...")
	
	simplified_tweets = []
	
	# Process file line by line to avoid loading everything into memory
	with open(input_file_path, 'r', encoding='utf-8', errors='replace') as file:
		for line in file:
			line = line.strip()
			if not line or not line.startswith('{'): 
				continue
				
			# Extract just the fields we need using targeted regex
			user_id_match = re.search(r"'user':\s*'(\d+)'", line)
			if not user_id_match:
				#some user ids don't have quotes, so try without
				user_id_match = re.search(r"'user':\s*(\d+)", line)
			tweet_id_match = re.search(r"'id':\s*'(\d+)'", line)
			connection_type_match = re.search(r"'connection_type':\s*'(\w+)'", line)
			connected_user= re.search(r"'connected_user':\s*'(\w+)'",line)
			connected_tweet= re.search(r"'connected_tweet':\s*'(\w+)'",line)
			
			if user_id_match and tweet_id_match and connection_type_match:
				# Create simplified tweet object
				simplified_tweet = {
					"user_id": user_id_match.group(1),
					"tweet_id": tweet_id_match.group(1),
					"connection_type": connection_type_match.group(1),
					"connected_user": connected_user.group(1),
					"connected_tweet": connected_tweet.group(1)
				}
				simplified_tweets.append(simplified_tweet)
			else:
				print(f"Skipping line due to missing fields: {line}")
	
	print(f"Extracted {len(simplified_tweets)} tweets with required fields")
	
	# Save the simplified tweets to a new JSON file
	with open(output_file_path, 'w', encoding='utf-8') as out_file:
		json.dump(simplified_tweets, out_file, indent=2)
	
	print(f"Saved simplified data to {output_file_path}")
	return len(simplified_tweets)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Converts a file of tweets (one tweet per line, each in the MongoDump pseudo-JSON format) to a JSON file containing only user_id, tweet_id, connected_user, connected_tweet, and connection_type")
	parser.add_argument("-i", "--input_file", required=True, help="Path to the input file containing tweets")
	parser.add_argument("-o", "--output_file", required=False, help="Path to save the filtered tweets (defaults to the input file with .json extension)")

	args = parser.parse_args()
	
	start_time = time.time()
	
	if not os.path.exists(args.input_file):
		print(f"Input file does not exist: {args.input_file}")
		exit(1)
	if not args.output_file:
		args.output_file = args.input_file.rsplit('.', 1)[0] + '.json'
	if os.path.exists(args.output_file):
		input(f"Output file exists: {args.output_file}; Confirm overwrite (Ctrl+C to cancel)?")
	count = extract_tweet_network_data(args.input_file, args.output_file)
	print(f"Done. Processed {count} tweets")
	elapsed_time = time.time() - start_time
	print(f"Total time taken: {humanize.naturaldelta(elapsed_time)}")
	print(f"Memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")