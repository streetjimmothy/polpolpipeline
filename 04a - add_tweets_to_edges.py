import time
import humanize
import psutil
import json
import os
import argparse
import networkx as nx

def load_tweets_from_json(file_paths):
	"""
	Load tweets from a JSON file.
	"""
	print(f"Loading tweets from {file_paths}...")
	#Loading tweets from filtered samples
	filtered_tweets = []
	i = 0 #could do a range thing below, but cbf
	for file_path in file_paths:
		if not os.path.exists(file_path):
			print(f"File does not exist: {file_path}")
			continue
		with open(file_path, 'r', encoding='utf-8') as f:
			try:
				filtered_tweets += json.load(f)
			except json.JSONDecodeError as e:
				print(f"Error decoding JSON from {file_path}: {e}")
				continue
	return filtered_tweets

def get_tweets_by_user(tweets_json):
	"""
	Get tweets grouped by user.
	"""
	print("Grouping tweets by user...")
	users_tweets = {}
	for tweet in tweets_json:
		user_id = tweet["user_id"]
		if user_id not in users_tweets:
			users_tweets[user_id] = []
		users_tweets[user_id].append(tweet)
	return users_tweets

def add_tweets_to_edges(nx_g, users_tweets, verbose=False):
	"""
	Add tweet IDs to edges in the NetworkX graph.
	"""
	print("Adding tweets to edges...")
	for node, attrs in nx_g.nodes(data=True):
		user_id = attrs.get('id')
		if user_id in users_tweets:
			for connected_user in nx_g.neighbors(node):
				for tweet in users_tweets[user_id]:
					if tweet["connected_user"] == nx_g.nodes[connected_user]['id']:
						if 'tweets' not in nx_g[node][connected_user]:
							nx_g[node][connected_user]['tweets'] = []
						nx_g[node][connected_user]['tweets'].append(tweet["tweet_id"])

def save_nx_graph(graph, filename):
	"""
	Save the NetworkX graph to a file.
	"""
	print(f"Saving graph to {filename}...")
	for node, attrs in nx_g.nodes(data=True):
		for connected_user in nx_g.neighbors(node):
			if 'tweets' in nx_g[node][connected_user]:
				nx_g[node][connected_user]['tweets'] = ';'.join(nx_g[node][connected_user]['tweets'])  
	nx.write_graphml(graph, filename)
	print("Graph saved successfully.")

#at one point community detection is igraph stripped the tweet ids from edges, this re-adds them
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="adds tweet ids to edges in a networkx graph")
	parser.add_argument("-i", "--input_file","--input_files", "--input_dir", nargs='+', required=True, help="Path to the input JSON file containing tweets (multiple files can be specified, they will be graphed together. A directory can also be specified, in which case all JSON files in that directory will be used)")
	parser.add_argument("-o", "--output_file","--output_files", nargs='+', required=True, help="Path to save the graph (if the graph already exists, it will just have the tweet ids added to the edges). Multiple graphs can be supplied, they will be processed seperately from the same input files")
	parser.add_argument("--verbose", action='store_true', help="Enable verbose output for debugging and progress tracking")

	output_file_base = ""
	args = parser.parse_args()

	start_time = time.time()
	if not args.input_file:
		print("No input file specified")
		exit(1)
	if len(args.input_file) > 1:
		print(f"Multiple input files specified: {args.input_file}")
	else:
		if os.path.isdir(args.input_file[0]):
			# If a directory is specified, get all JSON files in that directory
			args.input_file = [os.path.join(args.input_file[0], f) for f in os.listdir(args.input_file[0]) if f.endswith('.json')]
			if not args.input_file:
				print(f"No JSON files found in directory: {args.input_file[0]}")
				exit(1)
			print(f"Input directory specified, using files: {args.input_file}")
		else:
			print(f"Single input file specified: {args.input_file[0]}")

	if not args.output_file:
		print("No output file specified")
		exit(1)
	if len(args.output_file) > 1:
		print(f"Multiple output files specified: {args.output_file}")
	else:
		output_file = args.output_file
		print(f"Output file: {output_file}")

	tweets_json = load_tweets_from_json(args.input_file)
	users_tweets = get_tweets_by_user(tweets_json)
	for file_path in args.output_file:
		nx_g = nx.read_graphml(file_path)
		add_tweets_to_edges(nx_g, users_tweets, verbose=args.verbose)
		save_nx_graph(nx_g, file_path)
	
	elapsed_time = time.time() - start_time
	print(f"Total time taken: {humanize.naturaldelta(elapsed_time)}")
	print(f"Memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")