# import modules used for performance profiling
import time
import humanize
import psutil
import os
import argparse
import csv

# import modules used for network analysis
import igraph
import leidenalg

# MongoDB imports
from pymongo import MongoClient, errors
CONNECTION_STRING = "mongodb://JamIs:morticiaetpollito@118.138.244.29:27017/"

import utilities as util

def print_community_stats(community_graph, main_graph, min_major_community_size=5):
	# Get all subgraphs
	subgraphs = community_graph.subgraphs()
	# Sort subgraphs by size in descending order
	sorted_subgraphs = sorted(subgraphs, key=lambda x: len(x.vs), reverse=True)
	for i, subgraph in enumerate(sorted_subgraphs):
		if len(subgraph.vs)/len(main_graph.vs) > min_major_community_size / 100:
			print(f"Community {i}: {len(subgraph.vs)} nodes")
			print(f"Community {i} as a proportion of total: {(len(subgraph.vs)/len(main_graph.vs)):.2%}")

	# Filter subgraphs with less than 10 nodes
	small_subgraphs = [sg for sg in subgraphs if len(sg.vs) < 10]
	print("Small Graphs (<10 nodes): {}".format(len(small_subgraphs)))


def read_file(file_path):
	"""
	Read a graphml file and return the data.
	"""
	print(f"Reading file: {file_path}")
	try:
		graph = igraph.Graph.Read_GraphML(file_path)
		for edge in graph.es:
			# Convert 'tweets' attribute from string to list if it exists
			if 'tweets' in edge.attributes():
				tweet_ids = []
				if edge['tweets'] is not None:
					# Split the string by semicolon and remove empty strings
					# This handles cases where 'tweets' might be an empty string or None
					# and ensures we don't end up with an empty list if there are no tweets
					tweet_ids = edge['tweets'].split(";")
					tweet_ids = [tweet_id.strip() for tweet_id in tweet_ids if tweet_id.strip()] # Remove empty strings ## Is this necessary??
				edge['tweets'] = tweet_ids
		print(f"Successfully read graph from {file_path}")
		print("Nodes: {}".format(len(graph.vs)))
		print("Edges: {}".format(len(graph.es)))
		return graph
	except Exception as e:
		print(f"Error reading graph from {file_path}: {e}")
		return None


def run_community_detection(graph):
	print("Doing leidenalg...")
	start_time = time.time()
	ig_community_graph = leidenalg.find_partition(graph.connected_components("weak").giant(), leidenalg.ModularityVertexPartition, seed=5);
	print("Done. Graphs: {}".format(len(ig_community_graph.subgraphs())))
	for node in graph.vs():
		try:
			node['T'] = ig_community_graph.membership[node.index]
		except IndexError:
			# node not a part of the largest weakly connected component, so it won't have a community
			pass
	elapsed_time = time.time() - start_time
	print(f"Total time taken: {humanize.naturaldelta(elapsed_time)}")
	print(f"Memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
	return ig_community_graph


def save_file(output_file, graph):
	print(f"Saving community information to {output_file}...")
	try:
		for edge in graph.es:
			# Convert 'tweets' attribute from list to a string if it exists
			if 'tweets' in edge.attributes():
				if edge['tweets'] is not None:
					edge['tweets'] = ';'.join(edge['tweets'])  # Join the list into a string
		graph.write_graphml(output_file)
		print(f"Community information saved to {output_file}")
	except Exception as e:
		print(f"Error saving community graph to {output_file}: {e}")


def save_central_tweets(
	community_graph, 
	main_graph, 
	output_file, 
	num_nodes=0, 
	target_community=-1, 
	p=5
):
	# PRINT TWEETS FROM num_nodes MOST CENTRAL NODES FROM EACH COMMUNITY that is more than p% of the graph TO FILE

	# CONNECT TO DATABASE
	print("connecting...")
	client = MongoClient(CONNECTION_STRING)
	tw_coll = client.get_database('Tw_Covid_DB').get_collection('tweets')
	print("connected")
	# internal function to make a query to the database, breaking it into chunks if necessary

	def make_query(query_list):
		# Create a query to find tweets with the specified IDs
		try:
			print(f"Querying database for {len(query_list)} tweets...")
			cursor = tw_coll.find({"_id": {'$in': query_list}})
		except errors.DocumentTooLarge as e:
			# If the query is too large, split it into smaller chunks
			cursor = []
			for i in range(0, len(query_list), 1000):
				chunk = query_list[i:i + 1000]
				cursor.extend(tw_coll.find({"_id": {'$in': chunk}}))
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

	print("Collecting graph stats...")
	start_time = time.time()
	# Get all subgraphs
	subgraphs = community_graph.subgraphs()
	#TODO: Should be directed?
	all_betweenness = main_graph.betweenness()
	all_closeness = main_graph.closeness()
	all_eigenvector_centrality = main_graph.eigenvector_centrality()
	print(f"Total time taken: {humanize.naturaldelta(time.time() - start_time)}")
	print(f"Memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

	# Sort subgraphs by size in descending order
	sorted_subgraphs = sorted(subgraphs, key=lambda x: len(x.vs), reverse=True)
	for i, subgraph in enumerate(sorted_subgraphs):
		if len(subgraph.vs)/len(main_graph.vs) > p / 100 and (target_community == -1 or target_community == i):
			print ("Community ", i)
			#Find higher centrality nodes in each subgraph
			print("Collecting community stats...")
			start_time = time.time()
			nodes = sorted(subgraph.vs, key=lambda vertex: vertex.degree(), reverse=True)
			central_nodes = nodes[:num_nodes]
			comm_betweenness = subgraph.betweenness()
			comm_closeness = subgraph.closeness()
			comm_eigenvector_centrality = subgraph.eigenvector_centrality()
			print(f"Total time taken: {humanize.naturaldelta(time.time() - start_time)}")
			print(f"Memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
			if num_nodes == -1:
				filename = f"{output_file}_{i}comm_all"
			else:
				filename = f"{output_file}_{i}comm_{num_nodes}central"
			with \
				open(f"{filename}_nodes_tweets.csv", "w", encoding="utf-8") as csv_file, \
				open(f"{filename}_nodes_stats.csv", "w", encoding="utf-8") as nodes_csv_file, \
				open(f"{filename}_nodes_tweets.txt", "w", encoding="utf-8") as text_file:
					writer = csv.writer(csv_file)
					nodes_writer = csv.writer(nodes_csv_file)
					# Write header
					writer.writerow(["community_id", "user_id", "tweet_id", "original_user", "original_tweet_id", "tweet_text"])
					nodes_writer.writerow(["community_id", "user_id", "degree", 
						"comm_betweenness", "comm_closeness", "comm_eigenvector_centrality", 
						"all_betweenness", "all_closeness", "all_eigenvector_centrality"
					])

					#2 Get tweets from higher centrality nodes in each subgraph
					#get each edge for each author
					for node in central_nodes:
						#write the node stats to the file
						nodes_writer.writerow([i, node["id"], node.degree(), 
							comm_betweenness[node.index], comm_closeness[node.index], comm_eigenvector_centrality[node.index], 
							all_betweenness[node.index], all_closeness[node.index], all_eigenvector_centrality[node.index]
						])

						edges_ids = [] #list for tweets ids from igraph
						try: 
							for edge in node.incident(mode="all"): #get all edges per node (tweet per user)
								if 'tweets' in edge.attributes(): #if the edge has tweets
									for tweet_id in edge['tweets']:
										edges_ids.append(tweet_id) #puts in a list to match with db
						
							tweet_collection = make_query(edges_ids) #find tweets Id in DB
							root_tweets = [] #list to find the truncated root
							for tweet in tweet_collection:
								if tweet['connection_type'] and tweet['connection_type'] == "retweet": #if the tweet is a rt
									root_tweets.append(tweet['connected_tweet']) #then put the root id in the list
								else:
									writer.writerow([i, tweet['user'], tweet['_id'], tweet['connected_user'], tweet['connected_tweet'], tweet['text'].replace('\n', '   ')]) #if is an OG just write it
									text_file.write(tweet["text"].replace('\n', '   ')+'\n') 

							tweet_collection = make_query(root_tweets)  #find the roots of truncated
							for tweet in tweet_collection:
								writer.writerow([i, tweet['user'], tweet['_id'], "", "", tweet['text'].replace('\n', '   ')])
								text_file.write(tweet["text"].replace('\n', '   ')+'\n') 
						except Exception as e:
							print(len(edges_ids))
							print(len(root_tweets))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Runs community detection. Optionally saves central tweets to a file.")
	util.create_input_args(parser, help="Path to the input graphml file (multiple files can be specified, the process will be run for each and they will be saved to the specified output file - or, if no output file is specified each will be saved back to the input file. A directory can also be specified, in which case all graphml files in that directory will be used)")
	util.create_output_args(parser, suffix=".graphml (i.e. save back to input file)")
	
	parser.add_argument("--save-central-tweets", type=int, default=0, help="Save central tweets to a file (number of users to save tweets for, default: 0, which means no tweets will be saved. -1 will save all tweets in each community). Tweets will be saved to the output file with '_{n}central_{community_size_ranking}comm.csv' appended to the filename.")
	parser.add_argument("--community-size", type=int, default=5, help="Size of a community to consider, as a percentage of the total graph (default: 5%%). Communities smaller than this size will not be considered for central tweets extraction.")
	parser.add_argument("--target-community", type=int, default=-1, help="If specified, only save tweets from this community (default: -1, which means all communities will be processed). 0 indexed.")
	parser.add_argument("--verbose", action='store_true', help="Enable verbose output for debugging and progress tracking")
	parser.add_argument("-db", "--db",required=False, help="Database connection string override (default: connects to Nectar MongoDB instance)")
	#parser.add_argument("--follow_RTs", "--follow_rts", action='store_true', help="When saving central tweets, follow retweets to get the original tweet text.")

	args = parser.parse_args()

	# if args.verbose:
	# 	logging.basicConfig(level=logging.DEBUG, format='%(asctime)s::%(levelname)s:%(message)s')
	# else:
	# 	logging.basicConfig(level=logging.INFO, format='%(asctime)s::%(levelname)s:%(message)s')

	if args.db:
		CONNECTION_STRING = args.db

	start_time = time.time()
	input_files = util.parse_input_files_arg(args.input_file, ext=".graphml")
	output_files = util.parse_output_files_arg(args.output, input_files)

	for input_file, output_file in zip(input_files, output_files):
		if not os.path.exists(input_file):
			print(f"Input file does not exist: {input_file}")
			exit(1)
		ig_g = read_file(input_file)
		if ig_g is None:
			print(f"Skipping file {input_file} due to read error")
			continue
		print(f"Running community detection on {input_file}...")
		ig_community_graph = run_community_detection(ig_g)
		print_community_stats(ig_community_graph, ig_g, args.community_size)
		if output_file:
			save_file(output_file, ig_g)
		print(f"Community detection completed for {input_file}")
		print("Saving central tweets to file...")
		save_central_tweets(
			ig_community_graph, 
			ig_g, 
			output_file=os.path.splitext(output_file)[0], 
			num_nodes=args.save_central_tweets, 
			target_community=args.target_community, 
			p=args.community_size
		)


	elapsed_time = time.time() - start_time
	print(f"Total time taken: {humanize.naturaldelta(elapsed_time)}")
	print(f"Memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")