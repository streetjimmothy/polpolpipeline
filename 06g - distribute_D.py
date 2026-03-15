import argparse
import utilities as utils
import networkx as nx

def by_edges(G):
	community_D_histogram = {}
	community_sizes = {}
	for node_id in G.nodes:
		neighbors = G.predecessors(node_id)
		D = G.nodes[node_id]['D']
		comm = G.nodes[node_id]['T']
		if comm not in community_sizes:
			community_sizes[comm] = 0
		neighbor_communities = [G.nodes[neighbor]['T'] for neighbor in neighbors]
		if comm not in community_D_histogram:
			community_D_histogram[comm] = {}
		if D not in community_D_histogram[comm]:
			community_D_histogram[comm][D] = {}
		for community in neighbor_communities:
			if community not in community_D_histogram[comm][D]:
				community_D_histogram[comm][D][community] = 0
			community_D_histogram[comm][D][community] += 1
			community_sizes[comm] += 1	#TODO: wait, shouldn't this be edges, not nodes?

		for comm, D_histogram in community_D_histogram.items():
			do_print = False
			D_sizes = {}
			for D, neighbor_comm_histogram in D_histogram.items():
				if D not in D_sizes:
					D_sizes[D] = 0
				for neighbor_comm, count in neighbor_comm_histogram.items():
					D_sizes[D] += count
					if count > 1000:
						do_print = True

			if do_print:
				print(f"Community {comm}, size {community_sizes[comm]}:")
				for D, neighbor_comm_histogram in D_histogram.items():
					print(f"  D={D}:")
					for neighbor_comm, count in neighbor_comm_histogram.items():
						if count > 100:
							print(f"    Neighbor community {neighbor_comm}: {count} edges ({count / D_sizes[D]:.2%} of D={D} neighbors, {count / community_sizes[comm]:.2%} of community {comm})")


def by_nodes(G):
	community_sizes = {}
	community_D_totalnodes = {}
	community_D_internalnodes = {}
	#for each node
	#allocate it to a community bin and a D bin
	#for the neighbouring nodes, count
	for node_id in G.nodes:
		D = G.nodes[node_id]['D']
		comm = G.nodes[node_id]['T']
		if comm not in community_D_totalnodes:
			community_D_totalnodes[comm] = {}
		if D not in community_D_totalnodes[comm]:
			community_D_totalnodes[comm][D] = 0
		community_D_totalnodes[comm][D] += 1

		if comm not in community_D_internalnodes:
			community_D_internalnodes[comm] = {}
		if D not in community_D_internalnodes[comm]:
			community_D_internalnodes[comm][D] = 0
		neighbors = G.predecessors(node_id)
		neighbor_communities = [G.nodes[neighbor]['T'] for neighbor in neighbors]
		if sum(neighbor_comm == comm for neighbor_comm in neighbor_communities) > len(neighbor_communities) / 2:
			community_D_internalnodes[comm][D] += 1
		if comm not in community_sizes:
			community_sizes[comm] = 0
		community_sizes[comm] += 1
	
	#order the communities by size, and print the D distribution for each community, but only if the community is large enough and has a significant number of internal nodes for at least one D value
	community_D_internalnodes = dict(sorted(community_D_internalnodes.items(), key=lambda x: community_sizes[x[0]], reverse=True))
	for comm in community_D_internalnodes:
		community_D_internalnodes[comm] = dict(sorted(community_D_internalnodes[comm].items(), key=lambda x: x[0]))

	for comm, D_histogram in community_D_internalnodes.items():
		if community_sizes[comm] < 1000:
			continue
		print(f"Community {comm}, size {community_sizes[comm]}:")
		print("D \t % of D \t Internal nodes \t % of community")
		for D, internal_node_count in D_histogram.items():
			if internal_node_count > 0:
				print(
					f"{D}\t"
					f"{internal_node_count / community_D_totalnodes[comm][D]:.2%} \t"
					f"{internal_node_count}\t"
					f"{internal_node_count / community_sizes[comm]:.2%}"
				)	

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Prints D statistics for each community in a graphml file. GraphML file must havfe the WoC results embedded as node attributes.")
	utils.create_input_args(parser, ext=".graphml")

	args = parser.parse_args()

	input_paths = utils.parse_input_files_arg(args.input_file, ext=".graphml")

	
	for input_path in input_paths:
		print(f"Processing input file: {input_path}")
		G = nx.read_graphml(input_path)
		print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
		#by_edges(G)
		by_nodes(G)

