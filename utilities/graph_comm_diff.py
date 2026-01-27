import argparse
import networkx as nx

def load_graphml(path):
	return nx.read_graphml(path)

def get_communities_by_T(graph):
	communities = {}
	for node, data in graph.nodes(data=True):
		T = data.get('T')
		if T is not None:
			communities.setdefault(T, set()).add(node)
	# Order communities by size (descending)
	ordered = sorted(communities.values(), key=lambda x: len(x), reverse=True)
	return ordered

def main():
	parser = argparse.ArgumentParser(
		description="Compare top communities (by 'T' attribute) between two GraphML graphs and report overlap."
	)
	parser.add_argument("--g1", help="Path to first GraphML file")
	parser.add_argument("--g2", help="Path to second GraphML file")
	args = parser.parse_args()

	community_pairs = {2:3, 3:2}

	g1 = load_graphml(args.g1)
	g2 = load_graphml(args.g2)

	comms1 = get_communities_by_T(g1)
	comms2 = get_communities_by_T(g2)

	comm_len = min(len(comms1), len(comms2))
	print("Community,\t comm1 size, comm2 size, Common vertices, common % of smaller")
	for i in range(comm_len):
		c1 = comms1[i]
		if i in community_pairs:
			j = community_pairs[i]
			c2 = comms2[j]
		else:
			c2 = comms2[i]
		# Use the 'id' attribute for comparison if present, else use node label
		ids1 = set(g1.nodes[n].get('id', n) for n in c1)
		ids2 = set(g2.nodes[n].get('id', n) for n in c2)
		common = ids1 & ids2
		print(f"{i}\t{len(c1)}\t{len(c2)}\t{len(common)}\t{(len(common) / min(len(c1), len(c2))) * 100:.2f}%")

if __name__ == "__main__":
	main()