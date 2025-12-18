import sys
import networkx as nx

def main():
	if len(sys.argv) != 4:
		print("Usage: python get_path.py <graphml_file> <start_node_id> <end_node_id>")
		sys.exit(1)

	graphml_file = sys.argv[1]
	start_node_id = sys.argv[2]
	end_node_id = sys.argv[3]

	# Load the graph
	G = nx.read_graphml(graphml_file)

	# Find the shortest path
	try:
		for node, data in G.nodes(data=True):
			if data['id'] == start_node_id:
				start_node = node
			if data['id'] == end_node_id:
				end_node = node
		path = nx.shortest_path(G, source=start_node, target=end_node)
		print("Shortest path:", " -> ".join(path))
	except nx.NetworkXNoPath:
		print(f"No path found between {start_node} and {end_node}.")
	except nx.NodeNotFound as e:
		print(e)

if __name__ == "__main__":
	main()