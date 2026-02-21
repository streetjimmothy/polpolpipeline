
import time
import humanize
import psutil
import json
import os
import argparse
import utilities as utils
from tqdm import tqdm

#import modules used for network analysis
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

def create_nx_graph(tweets_json, connections_to_keep=["retweet", "reply", "quote"]):
	"""
	Create a NetworkX graph from the tweets.
	"""
	print("Creating NetworkX graph...")
	nx_g = nx.DiGraph()
	print("building networkx graph...")
	start_time = time.time()
	for tweet in tqdm(tweets_json, desc="Building graph"):
		if tweet["user_id"] not in nx_g:
			nx_g.add_node(tweet["user_id"])
		
		if tweet["connected_user"] not in nx_g:
			nx_g.add_node(tweet["connected_user"])
		
		if tweet["connected_user"] != tweet["user_id"] and tweet["connection_type"] in connections_to_keep:
			if tweet["user_id"] not in nx_g[tweet["connected_user"]]:
				nx_g.add_edge(tweet["connected_user"], tweet["user_id"], weight=1, tweets=[tweet["tweet_id"]])
			else:
				nx_g[tweet["connected_user"]][tweet["user_id"]]['weight'] += 1
				nx_g[tweet["connected_user"]][tweet["user_id"]]['tweets'].append(tweet["tweet_id"])
	end_time = time.time()
	print("Time taken: {}".format(humanize.precisedelta(end_time - start_time, suppress=['days', 'milliseconds', 'microseconds'])))
	print("Memory used: {}".format(humanize.naturalsize(psutil.Process().memory_info().rss)))
	print("Nodes: {}".format(nx_g.number_of_nodes()))
	print("Edges: {}".format(nx_g.number_of_edges()))
	print("networkx graph built")
	return nx_g

def save_nx_graph(graph, filename):
	"""
	Save the NetworkX graph to a file.
	"""
	print(f"Saving graph to {filename}...")
	if os.path.exists(filename):
		input(f"Output file exists: {filename}; Confirm overwrite (Ctrl+C to cancel)?")
	#before we save the graph to GraphML, we need to convert the tweets list to a string
	graph = nx.DiGraph(graph)  # Ensure we are working with a directed graph
	for u, v, data in graph.edges(data=True):
		tweet_ids = ""
		if 'tweets' in data and isinstance(data['tweets'], list):
			for tweet_id in data['tweets']:
				tweet_ids += str(tweet_id) + ";"
			data['tweets'] = tweet_ids[:-1]  # Remove the last semicolon
	nx.write_graphml(graph, filename)
	print(f"Graph saved to {filename}")

def prune_graph(graph, min_connections=2, min_weight=1, max_iterations=-1, connected_component_type="weak", verbose=True):
	#OUR IMPLEMENTATION OF woc.iteratively_prune_graph, 
	done = False
	iteration = 0
	threshold = min_connections #number connections
	weight_threshold = min_weight #minimum weight to be considered a connection

	nx_pruned = nx_g.copy()

	if verbose:
		print(f'[iteratively_prune_graph: threshold={threshold}, weight_threshold={weight_threshold}, connected_component_type={connected_component_type}, verbose.]')

	while not done:
		start_time = time.time()
		if max_iterations >= 0 and iteration >= max_iterations:
			if verbose:
				print(f"Max iterations reached: {max_iterations}. Stopping pruning.")
			break
		iteration += 1
		done = True
		if verbose:
			print(f'Iteration #{iteration}...')
			print(len(nx_pruned.nodes),len(nx_pruned.edges))
		nodes_to_cut = []

		# this part directly from paper
		# but accomodate directed and undirected graphs
		#nx_communities is guaranteed to be directed
		for node in tqdm(list(nx_pruned), desc=f"Pruning nodes (iter {iteration})"):
			i = nx_pruned.in_degree(node)
			o = nx_pruned.out_degree(node)
			if i + o <= threshold:
				nodes_to_cut.append(node)

		if len(nodes_to_cut) > 0:
			done = False
			nx_pruned.remove_nodes_from(nodes_to_cut)

		# then do the weighted-edge culling
		edges_to_cut = []
		for edge in tqdm(list(nx_pruned.edges), desc=f"Pruning edges (iter {iteration})"):
			try:
				if len(nx_pruned.edges[edge]["tweets"]) <= weight_threshold:
					edges_to_cut.append(edge)
			except KeyError:
				raise KeyError('Weight attribute for thresholding not present; failing.')

		if len(edges_to_cut) > 0:
			done = False
			nx_pruned.remove_edges_from(edges_to_cut)

		# now greatest connected component - here we DON'T SQUASH. We use the weakly connected graph (in prod, make this an option)
		if not done:
			if connected_component_type == "strong":
				nx_pruned = nx.DiGraph(nx_pruned.subgraph(sorted(nx.strongly_connected_components(nx_pruned), key=len, reverse=True)[0]))
			elif connected_component_type == "weak":
				nx_pruned = nx.DiGraph(nx_pruned.subgraph(sorted(nx.weakly_connected_components(nx_pruned), key=len, reverse=True)[0]))
		if verbose:
			print("Time taken: {}".format(humanize.precisedelta(time.time() - start_time, suppress=['days', 'milliseconds', 'microseconds'])))
			print("Memory used: {}".format(humanize.naturalsize(psutil.Process().memory_info().rss)))
	return nx_pruned

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="creates a graph from tweets in a JSON file")
	utils.create_input_args(parser, ext=".json")
	utils.create_output_args(parser, suffix="{full|{cc_type}_pruned}.graphml")
	parser.add_argument("--verbose", action='store_true', help="Enable verbose output for debugging and progress tracking")
	prune_group = parser.add_argument_group('Graph Buiding Options')
	parser.add_argument("--connection_types","--types", nargs='+', default=["retweet", "reply", "quote"], help="Types of connections to include in the graph (default: retweet, reply, quote)")
	# Create a group for pruning-related arguments
	prune_group = parser.add_argument_group('Pruning Options')
	prune_group.add_argument("--min_connections","--connections", type=int, default=2, help="Minimum number of connections for a node to be kept in the graph (default: 2)")
	prune_group.add_argument("--min_weight","--weight", type=int, default=1, help="Minimum weight for an edge to be kept in the graph (default: 1)")
	prune_group.add_argument("--max_iterations","--iterations", type=int, default=-1, help="Maximum number of iterations for pruning the graph (-1 for no limit, default: -1)")
	prune_group.add_argument("--cc_type", default="weak", help="Whether to take the weak or strongly connected componented when pruning (default: weak)")

	output_file_base = ""
	args = parser.parse_args()

	start_time = time.time()

	input_files = utils.parse_input_files_arg(args.input_file, ext=".json")
	print(f"Output file base: {args.output}")

	tweets_json = load_tweets_from_json(input_files)
	nx_g = create_nx_graph(tweets_json, connections_to_keep=args.connection_types)
	save_nx_graph(nx_g, args.output + "_full.graphml")
	nx_pruned = prune_graph(nx_g, args.min_connections, args.min_weight, args.max_iterations, args.cc_type, args.verbose)
	save_nx_graph(nx_pruned, f"{args.output}_{args.cc_type}-pruned.graphml")

	elapsed_time = time.time() - start_time
	print(f"Total time taken: {humanize.naturaldelta(elapsed_time)}")
	print(f"Memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")