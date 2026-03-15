import argparse
import time
import tqdm
import csv
import networkx as nx
from dataclasses import dataclass
import utilities as utils


@dataclass(slots=True, frozen=True, order=True)
class NodeData:
	community: int
	S: int
	D: int
	π: int

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Plots an existing WoC output from a csv. \n Saves output to a sullivan plot")
	utils.create_input_args(parser, ext=".csv")
	utils.create_output_args(parser, suffix=".graphml")
	parser.add_argument("--verbose", action='store_true', help="If set, prints additional information during processing.")

	args = parser.parse_args()

	input_paths = utils.parse_input_files_arg(args.input_file, ext=".csv")
	output_paths = utils.parse_output_files_arg(args.output, input_paths)

	for input_path, output_path in zip(input_paths, output_paths):
		print(f"Processing input file: {input_path}")
		print(f"Output will be saved to: {output_path}")

		nodes = {}  # maps node_id -> NodeData(community, S, D, π)
		with open(input_path, 'r') as f:
			reader = csv.DictReader(f)
			for row in tqdm.tqdm(reader, desc="Reading WoC results", unit=" rows", disable=not args.verbose):
				_S = int(row['S'])
				_D = int(row['D'])
				_π = _S * _D
				nodes[int(row['Vertex'])] = NodeData(
					community=0,
					S=_S,
					D=_D,
					π=_π
				)

		G = nx.read_graphml(output_path)
		for node_id, node_data in tqdm.tqdm(nodes.items(), desc="Updating graph", unit="node", disable=not args.verbose):
			nx_node_id = f"n{node_id}"
			if nx_node_id in G:
				G.nodes[nx_node_id]['community'] = node_data.community
				G.nodes[nx_node_id]['S'] = node_data.S
				G.nodes[nx_node_id]['D'] = node_data.D
				G.nodes[nx_node_id]['π'] = node_data.π
			else:
				print(f"Warning: Node {node_id} from WoC results not found in graph. Skipping.")
		nx.write_graphml(G, output_path)