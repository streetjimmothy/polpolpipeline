
import argparse
import networkx as nx
from collections import defaultdict
from dataclasses import dataclass
from csv import writer

# MongoDB imports
from pymongo import MongoClient, errors
CONNECTION_STRING = "mongodb://JamIs:morticiaetpollito@118.138.244.29:27017/"


def reverse_graph_edges(G):
	return G.reverse(copy=True)

def run_pagerank(G):
	return nx.pagerank(G)

@dataclass
class Node:
	number: int
	community: str
	pagerank: float
	id: str
	#to db filled from database
	name: str | None = None
	screen_name: str | None = None
	url: str | None = None
	description: str | None = None


def get_top_nodes_by_community(G, pagerank_scores, community_attr='community', top_n=5):
	# Group nodes by community
	communities = defaultdict(list)
	for node_id, data in G.nodes(data=True):
		community = data.get(community_attr, None)
		if community is not None:
			communities[community].append(
				Node(
					node_id, 
					community, 
					id=data.get("id", 0), 
					pagerank=pagerank_scores.get(node_id, 0)
				)
			)
	# For each community, sort nodes by pagerank
	top_nodes = {}
	for community, nodes in communities.items():
		ranked = sorted(nodes, key=lambda n: pagerank_scores.get(n.number, 0), reverse=True)
		top_nodes[community] = ranked[:top_n]
	return top_nodes


def collect_node_data_from_db(nodes):
	try:
		client = MongoClient(CONNECTION_STRING)
		u_coll = client.get_database('Tw_Covid_DB').get_collection('users')
	except errors.PyMongoError as e:
		print(f"Error connecting to MongoDB: {e}")
		exit()

	def make_query(query_list):
		# Create a query to find tweets with the specified IDs
		try:
			print(f"Querying database for {len(query_list)} users...")
			cursor = u_coll.find({"_id": {'$in': query_list}})
		except errors.DocumentTooLarge as e:
			# If the query is too large, split it into smaller chunks
			cursor = []
			for i in range(0, len(query_list), 1000):
				chunk = query_list[i:i + 1000]
				cursor.extend(u_coll.find({"_id": {'$in': chunk}}))
			cursor = list(cursor)
		except errors.OperationFailure as e:
			print(f"Operation failed: {e}")
			exit()
		except errors.ServerSelectionTimeoutError as e:
			print(f"Server selection timeout: {e}")
			exit()
		except errors.NetworkTimeout as e:
			print(f"Network timeout: {e}")
			exit()
		except errors.CursorNotFound as e:
			print(f"Cursor not found: {e}")
			exit()
		except errors.ExecutionTimeout as e:
			print(f"Execution timeout: {e}")
			exit()
		except errors.PyMongoError as e:
			print(f"General PyMongo error: {e}")
			exit()
		return cursor

	
	user_data = make_query([node.id for node in nodes])
	user_data_dict = {str(user['_id']): user for user in user_data}
	for node in nodes:
		user_info = user_data_dict.get(str(node.id), {})
		node.name = user_info.get('name', None)
		node.screen_name = user_info.get('screen_name', None)
		node.url = user_info.get('url', None)
		node.description = user_info.get('description', None)
	
	
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Reverse graph edges and run PageRank, then print top nodes by community.")
	parser.add_argument("-i", "--input_graphml", help="Input GraphML file path")
	parser.add_argument("-o", "--output", required=False, help="Output file path. Defaults to input file with '_RPR.csv' suffix.", default=None)
	parser.add_argument("-c", "--community-attr", default="community", help="Node attribute for community (default: community)")
	parser.add_argument("-n", "--top-n", type=int, default=5, help="Number of top nodes per community to display (default: 5)")
	parser.add_argument("-db", "--db", required=False, help="Database connection string override (default: connects to Nectar MongoDB instance)")
	args = parser.parse_args()

	if args.db:
		CONNECTION_STRING = args.db

	# Load graph
	G = nx.read_graphml(args.input_graphml)
	# Reverse edges
	G_rev = reverse_graph_edges(G)
	# Run PageRank
	pagerank_scores = run_pagerank(G_rev)
	# Get top nodes by community
	top_nodes = get_top_nodes_by_community(G_rev, pagerank_scores, args.community_attr, args.top_n)

	output_path = args.output
	if output_path is None:
		output_path = args.input_graphml.rsplit('.', 1)[0] + '_RPR.csv'

	# Collect additional node data from database
	for community, nodes in top_nodes.items():
		collect_node_data_from_db(nodes)

	with open(output_path, 'w', newline='', encoding="utf-8") as f:
		csv_writer = writer(f)
		csv_writer.writerow([
			"node",
			"community",
			"PageRank",
			"id",
			"name",
			"screen_name",
			"url",
			"description"
		])
		for community, nodes in top_nodes.items():
			for node in nodes:
				csv_writer.writerow([
					node.number,
					node.community,
					node.pagerank,
					node.id,
					node.name,
					node.screen_name,
					node.url,
					node.description
				])

	# Print results
	for community, nodes in top_nodes.items():
		print(f"Community: {community}")
		for node in nodes:
			print(f"  {node} (PageRank: {node.pagerank:.5f})")
		print()
