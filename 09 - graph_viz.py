#!/usr/bin/env python3
"""Read a graph from a GraphML file and render it to PNG using ForceAtlas2 layout."""
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def main():
	parser = argparse.ArgumentParser(description="Read a GraphML file and render it to PNG using ForceAtlas2 layout")
	parser.add_argument("-i", "--input_file",type=str,help="Path to the input GraphML file")
	parser.add_argument("-o", "--output",type=str,default=None,help="Output PNG file path (default: input_file with .png extension)")
	parser.add_argument("--size",type=int,default=12,	help="Figure size in inches (default: 12)")
	parser.add_argument("--dpi",type=int,default=150,help="DPI for output PNG (default: 150)")
	parser.add_argument("--iterations",type=int,default=100,help="ForceAtlas2 iterations (default: 100)")
	parser.add_argument("--node-size",type=int,default=300,help="Node size for visualization (default: 300)")
	parser.add_argument("--community-colours",type=str,required=True,help="Path to a json file mapping community labels to colours")

	args = parser.parse_args()

	input_path = Path(args.input_file)
	if not input_path.exists():
		print(f"Error: Input file not found: {input_path}")
		return 1

	# Determine output path
	if args.output:
		output_path = Path(args.output)
	else:
		output_path = input_path.with_suffix(".png")

	print(f"Loading graph from {input_path}...")
	G = nx.read_graphml(str(input_path))
	print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

	pos = nx.forceatlas2_layout(
		G,
		max_iter=args.iterations,
		backend="nx-cugraph" if "nx-cugraph" in nx.config.available_backends() else "networkx"
	)

	# Load community colours
	import json
	with open(args.community_colours, 'r') as f:
		community_info = json.load(f)
		community_colours = community_info.get("community_colours", "")
		community_labels = community_info.get("community_labels", "")




	# Draw
	print(f"Drawing graph to {output_path}...")
	fig, ax = plt.subplots(figsize=(args.size, args.size))

	attr_values = nx.get_node_attributes(G, args.community_label)

	communities_by_attr = defaultdict(set)
	for node, value in attr_values.items():
		communities_by_attr[value].add(node)

	communities = list(communities_by_attr.values())

	for i, community in enumerate(communities):
		subgraph = G.subgraph(community)
		nx.draw_networkx_nodes(
			subgraph, pos,
			node_size=args.node_size,
			node_color=community_colours.get(community_labels[i], "xkcd:lightgrey"),
			ax=ax
		)
	nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)

	# Draw labels if nodes have them
	if all("label" in G.nodes[node] for node in list(G.nodes())[:min(10, G.number_of_nodes())]):
		labels = {node: G.nodes[node].get("label", node) for node in G.nodes()}
		nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

	ax.axis("off")
	fig.tight_layout()
	fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
	print(f"Saved to {output_path}")
	plt.close(fig)

	return 0


if __name__ == "__main__":
	exit(main())
