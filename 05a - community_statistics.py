import argparse
import os
import networkx as nx
from collections import defaultdict
import utilities as utils

def print_community_stats(community_graph, main_graph, min_major_community_size=5):
	# Get all subgraphs
	threshold_size = (min_major_community_size / 100) * main_graph.number_of_nodes()
	large_communities = [c for c in community_graph if len(c) > threshold_size]
	large_communities.sort(key=lambda x: len(x), reverse=True)
	print(f"\nNumber of communities larger than {min_major_community_size}% of graph size: {len(large_communities)}")
	# Filter subgraphs with less than 10 nodes
	print("Small Graphs (<10 nodes): {}".format(len([c for c in community_graph if len(c) >= 10])))

	mod_score = nx.algorithms.community.modularity(main_graph, community_graph)
	print(f"\nModularity: {mod_score:.4f}")
	print(f"Average Degree: {sum(dict(main_graph.degree()).values()) / main_graph.number_of_nodes():.4f}")

	for idx, community in enumerate(large_communities):
		subgraph = main_graph.subgraph(community)
		num_nodes = subgraph.number_of_nodes()
		num_edges = subgraph.number_of_edges()
		density = nx.density(subgraph)
		try:
			avg_clustering = nx.average_clustering(subgraph.to_undirected())
		except:
			avg_clustering = 0.0
		print(f"\nCommunity {idx}:")
		print(f" Size (nodes): {num_nodes}")
		print(f" % of total graph: {(num_nodes / main_graph.number_of_nodes()) * 100:.2f}%")
		print(f" Size (edges): {num_edges}")
		print(f" Density: {density:.4f}")
		print(f" Average Clustering Coefficient: {avg_clustering:.4f}")



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Prints statistics about communities in a graphml file.")
	utils.create_input_args(parser, ext=".graphml")
	parser.add_argument("--verbose", action='store_true', help="Enable verbose output for debugging and progress tracking")
	parser.add_argument("--community_label", "-C", help="Label attribute for community detection (default: 'community')", default="community")
	parser.add_argument("--community-size", type=int, default=5, help="Size of a community to consider, as a percentage of the total graph (default: 5%).")

	args = parser.parse_args()

	input_files = utils.parse_input_files_arg(args.input_file, ext=".graphml")

	for input_path in input_files:
		if not os.path.exists(input_path):
			print(f"Input file does not exist: {input_path}")
			exit(1)
		if args.verbose:
			print(f"Loading graph from: {input_path}")
		G = nx.read_graphml(input_path)
		if args.verbose:
			print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
		attr_values = nx.get_node_attributes(G, args.community_label)

		communities_by_attr = defaultdict(set)
		for node, value in attr_values.items():
			communities_by_attr[value].add(node)

		communities = list(communities_by_attr.values())
		if args.verbose:
			print(f"Detected {len(communities)} communities based on attribute '{args.community_label}'.")
		print_community_stats(communities, G, min_major_community_size=args.community_size)

	