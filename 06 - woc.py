#import modules used for performance profiling
import time
import humanize
import psutil
import os
import argparse
import csv

#import modules used for network analysis
import networkx as nx

#import modules used for WOC analysis
import numpy as np
from collections import Counter, defaultdict

import itertools
from networkx.exception import NetworkXNoPath
#import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import logging

import sys
spinner = ['|', '/', '-', '\\']

import utilities as utils


class Crowd:
	"""
	Class for encapsulating a graph and pre-computed (memoized) features for the
	Wisdom of Crowds algorithm, per Sullivan et al. (2020),
	
	"Vulnerability in Social Epistemic Networks" *International Journal of Philosophical Studies*
	https://doi.org/10.1080/09672559.2020.1782562
	
	Attributes:
		G: networkx graph (see __init__)
		min_k: smallest k to consider during processing, defaults to 2
		max_k: largest k to consider during processing, defaults to 5
		min_m: smallest m to consider during processing, defaults to 1
		max_m: largest m to consider during processing
		node_key: attribute to consider for each node (see __init__)		
		precomputed_path_dict: cache for unconditional paths
		precomputed_paths_by_hole_node: cache for dict of paths per node
		refresh_requested: flag indicating if cache has expired
		node_set: a snapshot of nodes to detect cache expiry
		edge_set: a snapshot of nodes to detect cache expiry
		s_cache: cached versions of S results
	"""
	def __init__(self, G, max_m=5, node_key='T'):
		"""
		Constructor:
			`__init__`: Inits the Crowd object.
		
		Args:
			G: a networkx graph, typically DiGraph.
			max_m: maximum m to consider in the calculations
			node_key: attribute to consider for each node, when considering topic diversity (defaults to 'T')
		"""
		# object check to avoid null ptr reference
		if G is None:
			raise ValueError('Crowd: init requires a valid networkx graph!')
		self.G = G
		self.min_k = 2
		self.max_k = 5
		self.min_m = 1
		self.max_m = max_m
		self.node_key = node_key
		self.precomputed_path_dict = {} # holds unconditional paths
		self.precomputed_paths_by_hole_node = defaultdict(dict)  # holds dict of paths per node
		self.refresh_requested = False

		# if G is okay, we 'hash' the graph data to prevent external updates breaking internal caches
		# NB: weisfeiler_lehman_graph_hash(G) is the best, but is very performance-draining
		self.node_set = set(G.nodes())
		self.edge_set = set(G.edges())
		#cache S and St values too. This speeds up pi, and recalcs. cleared if you clear path dict.
		self.s_cache = {}
		self.t_cache = {}

		logger = logging.getLogger(__name__) #introduce a (default) logger for the class instance


	def __efficient_pairs(self, x):
		"""
		__efficient_pairs: internal function, makes search for possible cliques more efficient.
		
		This should not be called directly by the user.
		
		Args:
			x: input list
			
		Returns:
		  unique pairs of elements (x_i, x_j)
		"""
		l = len(x)
		for i in range(1, l):
			for j in range(0, i):
				yield (x[i], x[j])


	def __shortest_path_node_source_target(self, v, source, target):
		"""
		__shortest_path_node_source_target: internal function,
			to get the length of the shortest path between vertices source and target, excluding v.
			
		The results need to be postprocessed by wrapper function
		shortest_path_length_node_source_target and not called directly.
		This function does memoization for efficient future processing:
		for a single node, and a single source, precompute the path,
		and populate the current dictionary precomputed_path_dict with them.
		
		Args:
			v: vertex under exclusion
			source: source node
			target: target node
		
		Returns:
			integer z, in range 0 <= z < +infinity (unadjusted)
		"""
		# error checking: 'v' needs to exist.
		# (missing 'source' and 'target' raised by nx)
		if v not in nx.nodes(self.G):
			raise nx.NodeNotFound

		# step 1: am I in the generic path dictionary? (memoized)
		try:
			shortest_unconditional_path = self.precomputed_path_dict[(source,target)]
		except KeyError: #well figure this later in case it comes in handy
			try:
				shortest_unconditional_path = nx.algorithms.shortest_path(self.G,source,target)
				self.precomputed_path_dict[(source,target)] = shortest_unconditional_path
			except NetworkXNoPath:
				shortest_unconditional_path = []
				self.precomputed_path_dict[(source,target)] = shortest_unconditional_path
				for x in range(1,len(shortest_unconditional_path)-1):
					z = shortest_unconditional_path[x:]
					self.precomputed_path_dict[(z[1],target)] = z

		# step 2 check if this is also a path without the node of interest
		if v not in shortest_unconditional_path:
			return shortest_unconditional_path
		# now we have to find the shortest path in a subgraph without the node of interest
		else:
			try:
				shortest_conditional_path = self.precomputed_paths_by_hole_node[v][(source,target)]
				return shortest_conditional_path
			except KeyError:
				nodes_less_v = self.node_set - set([v])
				G_sub = self.G.subgraph(nodes_less_v)

				if not (source in G_sub and target in G_sub):
					# no path, as it doesn't exist anymore in the culled subgraph
					self.precomputed_paths_by_hole_node[v][(source,target)] = []
					return []
				try:
					shortest_conditional_path = nx.algorithms.shortest_path(G_sub,source,target)
					# note that subpaths could also be cached as per above
					self.precomputed_paths_by_hole_node[v][(source,target)] = shortest_conditional_path
					return shortest_conditional_path
				except NetworkXNoPath:
					self.precomputed_paths_by_hole_node[v][(source,target)] = []
					return []


	def shortest_path_length_node_source_target(self, v, source, target):
		"""
		shortest_path_length_node_source_target: wrapper function to get the length of the
			shortest path between vertices source and target, without vertex v.
			
			no path = infinite length
			
		Args:
			v: vertex under consideration, as defined by (Sullivan et al., 2020)
			source: source node
			target: target node
		
		Returns: 
			integer z, in range 0 <= z <= +infinity
		"""
		z = len(self.__shortest_path_node_source_target(v, source, target))
		if z == 0:
			return(float('inf'))
		else:
			#z-1 because the path is a list of nodes incl start and end ; we're using distance=number of edges, which is 1 less.
			return z - 1


	def is_mk_observer(self, v, m, k, transmit=False):
		"""
		is_mk_observer: checks if the vertex v is an (m,k)-observer as defined by (Sullivan et al., 2020);
			optimized clique-finding algo by CVK.
		
		If transmit=True, runs the algorithm but checks the node's position as a transmitter.
		This has the same result as running the algorithm on the reverse of the graph, 
			but is much more efficient due to not having redo the calculations for the reversed graph.
		
		Args:
			v	   : vertex to evaluate
			m	   : m as defined in (Sullivan et al., 2020); m >= 1
			k	   : k as defined in (Sullivan et al., 2020); k > 1
			transmit: boolean (defaults False)
		
		Returns:
			a boolean indicating the m,k-observer status of v
		"""
		if m < 1 or k <= 1:
			raise ValueError('Crowd: m needs to be integer >= 1; k needs to be integer > 1.')

		# PRECONDITION 1: if original graph seems to be 'obsolete',
		if set(nx.nodes(self.G)) != self.node_set or set(nx.edges(self.G)) != self.edge_set:
			# and PRECONDITION 2: AND ONLY IF the user fails to call clear_path_dict...
			if not self.refresh_requested:
				# throw error and hint as to how user can fix this by regenerating all intermediate data
				logging.warning('Performance warning: modifying G externally will result in "cache misses"; please refactor your code to avoid external modification, and to handle LookupErrors.')
				raise LookupError('Crowd: graph G has been modified externally, cached precomputed_path_dict is obsolete and need to be regenerated! Suggest using crowd.clear_path_dict()')
			else:
				# rehash the nodeset and edgeset so the graph is no longer detected as "changed"
				# i.e. on next run, the graph is considered "stable" and there is no need to request a refresh
				self.node_set = set(self.G.nodes())
				self.edge_set = set(self.G.edges())

				# user has confirmed that the cache has indeed been cleared.
				assert self.precomputed_path_dict == {}
				assert len(self.precomputed_paths_by_hole_node) == 0
				# disable the error detector for future runs (until the graph is tampered-with, again)
				self.refresh_requested = False

		if self.G.is_directed(): #this code snippet determines which nodes we measure the distance between, i.e. whether we measure v as an observer or transmitter
			if transmit==True:
				source_nodes = list(self.G.successors(v))
			else:
				source_nodes = list(self.G.predecessors(v))
		else:
			if transmit==True: logging.warning("Asked to check the position as transmitter for node", v, ", but since the graph is undirected this is redundant")
			source_nodes = list(self.G.neighbors(v))

		# if you have fewer than k, then you can't hear from at least k
		if len(source_nodes) < k:
			return False

		# special case, to ensure that a node with one input is a 1,1 observer
		if (len(source_nodes) == 1) and k==1 and m==1:
			return True

		max_k_found = False
		clique_dict = defaultdict(list) # this will get used to look for cliques

		# helper method __efficient_pairs makes sure that cliques are found and
		# early termination happens as soon as possible
		for source_a,source_b in self.__efficient_pairs(source_nodes):
			a_path_length = self.shortest_path_length_node_source_target(v,source_a,source_b)
			b_path_length = self.shortest_path_length_node_source_target(v,source_b,source_a)

			# if shortest path is too short, keep looking
			if (a_path_length<m) or (b_path_length<m):
				pass

			else:  # now we do the clique updating
				# first each pair trivially forms a clique
				# pairs are unique so we don't have to double-check as we go (i hope!)

				# first, this check is needed because if k<=2 then any hit at all satisfies it;
				# and it's time to go home
				if k<=2:
					return True

				trivial_clique = set([source_a,source_b])
				clique_dict[source_a].append(trivial_clique)
				clique_dict[source_b].append(trivial_clique)


			# now, for each pair of cliques for the two nodes, we have a new clique iff:
			# each clique has the same size m
			# the set containing the union of nodes from the two pairs of m-sized cliques is size m+1
			# so check the cliques in the nodes connected by the new pair
				for a, b in itertools.product(clique_dict[source_a], clique_dict[source_b]):
					lena = len(a)
					lenb = len(b)
					if lena != lenb:
						pass
					# avoid double counting
					# thogh you can probably do this faster by not adding the trivial clique until later?
					elif (a == trivial_clique) or (b == trivial_clique):
						pass
					else:
						node_union = a | b
						lenu = len(node_union)
						if lenu == (lena + 1):
							if lenu >= k:  # early termination
								max_k_found = True
								return max_k_found
							else:
								for node in node_union:
									clique_dict[node].append(node_union)
		return max_k_found


	def S(self, v, mk_tuple = False, transmit = False):
		"""
		S: calculates S, defined in (Sullivan et al., 2020) as the structural position of v. 
		If transmit == True, instead calculates calculates T, the inverse of S, i.e. the structural position of v as a transmitter.
		If mk == True, instead returns a tuple (S, m, k).
		To speed up future calculations, it also caches the result (S, m, k) in the s_cache dictionary.
		
			S = max_{(m,k) in MK}(m * k)
			T = max_{(m,k) in MK}(m * k), but running is_m_k_observer() with transmit = True
					
		Args:
			v:		  vertex to evaluate
			transmit:   whether to calculate the position as transmitter
			
		Returns:
			integer S or T, in range 0 <= (class constant max_m * class constant max_k)
			or tuple (S, m, k)
		"""

		try: #Attempt to retrieve previous results. If none exist, calculate S or T
			if transmit:
				t_c = self.t_cache[v]
				if mk:
					return t_c
				else:
					return t_c[0]
			else:
				s_c = self.s_cache[v]
				if mk:
					return s_c
				else:
					return s_c[0]
		except KeyError:
			pass

		possibilities = sorted([(m*k, m, k) for m, k in \
			itertools.product(range(self.min_m, self.max_m+1), \
							  range(self.min_k, self.max_k+1))], \
			reverse=True)

		for mk, m, k in possibilities:
			mk_observer = self.is_mk_observer(v, m, k, transmit=transmit)
			if mk_observer:
				if mk_tuple:
					if transmit:
						self.t_cache[v]= (mk, m, k)
					else:
						self.s_cache[v] = (mk, m, k)
					return (mk, m, k)
				else:
					return mk
			else:
				pass

		if transmit:
			self.t_cache[v]= (0,0,0)
		else:
			self.s_cache[v] = (0,0,0)
		if mk_tuple:
			return (0,0,0)
		else:
			return 0

	def D_edge(self, v, depth=None, selection=None):
		"""
		D_edge: calculates D edge-wise by seeing which topics are transmitted by the 
			informants of vertex v per (Sullivan et al. 2020)
		
		Args:
			param v			: vertex to evaluate
			param depth		: if we want to look past the immediate soures, how far back to look
			param selection	: if we want to only look at a selection of sources, these are the ones
		
		Returns:
			integer D_edge, in range 0 <= total topics attested in graph
		"""

		if selection != None:
			if v not in selection:
				selection = set(selection)
				selection.add(v)
			Gf = self.G.subgraph(nodes=selection) #the graph as it is used in this function (to allow for only looking at selections of sources)
		else:
			Gf = self.G.copy(as_view=True)
				  
		topics = s_topic = set()

		if depth != None:
			for s,t in nx.bfs_predecessors(nx.reverse(Gf), source=v, depth_limit=depth): #breadth-first search across nodes on a reversed graph to go upstream
				s_topic = self.G.nodes(data=self.node_key, default=None)[s]
		else:
			for e in Gf.in_edges(nbunch=v):
				s_topic = self.G.nodes(data=self.node_key, default=None)[e[0]]
		
		if s_topic is not None:
			if type(s_topic) is not set:
				topics.add(s_topic)
			else:
				topics.update(s_topic)
		return len(topics)

	def D(self, v):
		"""
		D: calculates D, defined in the literature as the number of topics found for
			informants of vertex v per (Sullivan et al., 2020)
			
			We apply the general case D = D' = | union_{(u,v) in E} C'(u) |
			
		Args:
			v: vertex to evaluate
		
		Returns:
			integer D, in range 0 <= D
		"""
		topics = set()
		source_nodes = self.G.predecessors(v)

		for s in source_nodes:
			s_topic = self.G.nodes(data=self.node_key, default=None)[s]
			if type(s_topic) is not set:
				topics.add(s_topic)
			else:
				topics.update(s_topic)
		return len(topics)


	def pi(self, v, transmit = False):
		"""
		pi: calculates pi, given vertex v, defined in (Sullivan et al., 2020) as the product of S and D
		
		Args:
			v: vertex to evaluate
			
		Returns:
			integer pi, where pi = S * D
		"""

		return self.D(v) * self.S(v, transmit=transmit)

	def h_measure(self, v, max_h=6, transmit = False):
		"""
		h_measure: find the highest h, given vertex v, of which mk_observer(v, h, h) is true
		
		Args:
			v: vertex to evaluate
			max_h: maximum_h to evaluate, defaults to 6 per (Sullivan et al., 2020)
		
		Returns:
			integer h, in range 1 < h <= max_h
		"""
		s,m,k = self.S(v, mk=True, transmit=transmit)
		return min(m,k)
		
		#for h in range(max_h, 1, -1): # recall (k > 1)
		#	if self.is_mk_observer(v, h, h, transmit):
		#		return h
		#
		# return 0
	
	def census(self, nbunch = None, topics = False):
		"""
		census: outputs a data structure containing the WoC network centrality measures for the nodes in the network (by default, all the nodes).
		Can be specified to also include the values about the topics the node transmits and receives about. 

		Args:
			nbunch: if specified, will only return the values for these nodes, takes a list, set, graph, etc.
			topics: Boolean which, if True, makes the function also output the measures about topics (D, pi, pi_t).
		
		Returns:
			dict output, with dictionaries of the WoC values keyed by node
		"""
		if nbunch is not None: #sets the graph as used by this function to either be the whole graph or (if specified) only a selection
			Gf = self.G.subgraph(nodes=nbunch) #the graph as it is used in this function (to allow for only looking at selections of sources)
		else:
			Gf = self.G.copy(as_view=True)
		
		output = dict()
		
		for n in Gf:
			output.update({n : dict( S=self.S(n, mk=True), St=self.S(n, mk=True, transmit=True), H=self.h_measure(n), Ht=self.h_measure(n, transmit=True) )})
		
		if topics:
			empty = True
			node_key = self.node_key
			for n in Gf:
				output[n].update({node_key:Gf.nodes(data=self.node_key, default=None)[n]})
				output[n].update({'D':self.D(n), 'pi':self.pi(n), 'pi_t':self.pi(n, transmit=True)})
	
		return output

	def clear_path_dict(self):
		"""
		clear_path_dict: helper function to completely reset the precomputed path dictionary.
		Should be used if G is changed.
		"""
		self.precomputed_path_dict = {}
		self.precomputed_paths_by_hole_node = defaultdict(dict)
		self.s_cache = {}
		self.refresh_requested = True
		return


"""
Now we add some additional utility/helper functions that are public-facing
These can be called by importing them
	from wisdom_of_crowds import make_sullivanplot
	from wisdom_of_crowds import iteratively_prune_graph
"""

def make_sullivanplot(pis, ds, ses, colormap='gist_yarg', suptitle=None, cax=None, yscale='linear', filename=None):
	"""
	make_sullivanplot: This makes the style of plot from Sullivan et al (2020).
	
	cvk note: Could be more generic, but essentially has two modes:
		
	* One, you can just pass a list of pis, Ds, and Ses, optionally with a colormap and a subtitle.
	  This will make and render a plot
	  
	* Two, or else you can pass an axis (and optionally colormap and suptitle)
	  and this will render it on the axis, allowing for multiple plots (as done in the paper figures).

	Args:
		pis: a list of pi-s
		ds:  a list of D-s
		ses: a list of S-s
		colormap: (optional) name of a colormap, defaults to 'gist_yarg'
		suptitle: (optional) supplementary title
		cax:  (optional) axis to render on
		yscale: (optional) scale of y-axis. Defaults to linear.
	
	Precondition:
		PRECONDITION: len(pis) == len(Ds) == len(Ses) == X, where len(X) > 0

	Returns:
		None on success; but generates the plot in a plt window.
	"""
	assert(len(pis) == len(ds) == len(ses))
	assert(len(pis) > 0)

	cmap = plt.get_cmap(colormap)
	norm = Normalize(vmin=min(ds)-1,vmax=max(ds)+1)

	# sort by pi, then d
	z = sorted([(pi,d,s) for pi,d,s in zip(pis,ds,ses)])
	pis = [pi for pi,d,s in z]
	sds = [(s,d) for pi,d,s in z]

	# make the pi values first
	total = len(pis)
	c = Counter(pis)
	cumulative = 0

	xs = [0]
	ys = [0]

	for pi in c:
		xs.append(cumulative)
		ys.append(ys[-1])
		xs.append(cumulative)
		ys.append(pi)
		cumulative += c[pi] / total


	# now build up the bar graph
	sdcounter = Counter(sds)
	total = len(pis)
	current_x = 0
	cumulative = 0

	barx = []
	barwidth = []
	barheight = []
	barcolors = []
	seen = []
	for pi,d,s in z:
		if (pi,s,d) in seen:
			pass
		else:
			barx.append(current_x)

			cumulative += (sdcounter[(s,d)]/total)
			current_x = cumulative

			barwidth.append(sdcounter[(s,d)]/total)
			barheight.append(s)
			barcolors.append(cmap(norm(d)))
			seen.append((pi,s,d))

	# do the plot
	if cax == None:
		fig = plt.figure(figsize=(12,6),facecolor='w')
		ax = fig.add_subplot(111)
	else:
		ax = cax

	ax.bar(barx,barheight,width=barwidth,color=barcolors,align='edge')
	ax.plot(xs,ys,c='k')

	ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
	ax.set_xlim((0,1))

	ax.yaxis.tick_right()
	ax.yaxis.grid()

	# make the legend for D
	handles = []
	for d in set(ds):
		handles.append(mpatches.Patch(color=cmap(norm(d)), label="D="+str(d)))
	ax.legend(handles=handles,loc='upper left')

	ax.set_yscale(yscale)

	if suptitle is not None:
		ax.set_title(suptitle)

	if cax==None:
		plt.savefig(filename if filename is not None else 'sullivan_plot.png', dpi=300, bbox_inches='tight')
		return None
		ax.set_xlabel('Proportion')
		ax.set_ylabel('S/pi')
	else:
		return None
	

#do wisdom of the crowds analysis
def wisdom_of_crowds(nx_graph, community=None, plot=False, crowd=None, filename=None, verbose=False):
	print("wisdom of the crowds:")
	if crowd is None:
		print("constructing crowd")
		crowd = Crowd(nx_graph)
		print("crowd constructed")
	else:
		print("using cached crowd")
	print("creating parameter sets")
	start_time = time.time()
	s_set = []
	d_set = []
	if verbose:
		i = 0
	for i, node in enumerate(crowd.node_set):
		if community == None:
			s_set.append(crowd.S(node))
			d_set.append(crowd.D(node))
		elif nx_graph.nodes[node]['T'] == community:
			s_set.append(crowd.S(node))
			d_set.append(crowd.D(node))
		if verbose:
			sys.stdout.write('\r' + spinner[i % len(spinner)] + ' Processing... ' + str(i) + ' of ' + str(len(crowd.node_set)))
			sys.stdout.flush()
			i += 1
	if verbose:
		sys.stdout.write('\rDone!                  \n')
	s_set = np.array(s_set)
	d_set = np.array(d_set)
	π_set = np.multiply(s_set,d_set)
	if verbose:
		print("s_set distribution:")
		# count the occurrences of each integer
		counter = Counter(s_set)
		print(counter)
		print("d_set distribution:")
		# count the occurrences of each integer
		counter = Counter(d_set)
		print(counter)
		print("π_set distribution:")
		# count the occurrences of each integer
		counter = Counter(π_set)
		print(counter)
		
	print("Time taken: {}".format(humanize.precisedelta(time.time() - start_time, suppress=['days', 'milliseconds', 'microseconds'])))
	print("Memory used: {}".format(humanize.naturalsize(psutil.Process().memory_info().rss)))
	print("parameter sets created")
	if plot:
		if community is not None:
			title = f"WoC analysis of community {community}"
			fn = f'_woc_community_{community}.png' if filename is None else filename + f'_woc_community_{community}.png'
		else:
			title = "WoC analysis of full graph"
			fn = 'woc_full_graph.png' if filename is None else filename + '_woc_full_graph.png'
		print("plotting")
		make_sullivanplot(π_set,d_set,s_set,colormap='magma_r', suptitle=title, filename=fn)
		print("plotting complete")
	if filename is not None:
		if community is not None:
			filename = f"{filename}_woc_community_{int(community)}.csv"
		else:
			filename = f"{filename}_woc.csv"
		print(f"saving results to {filename}")
		with open(filename, 'a', newline='') as csvfile:
			writer = csv.writer(csvfile)
			if not os.path.exists(filename) or os.stat(filename).st_size == 0:
				writer.writerow(['node','id','community','S','D','pi'])
			for node, s, d, pi in zip(crowd.node_set, s_set, d_set, π_set):
				writer.writerow([node,nx_graph.nodes[node]['id'],nx_graph.nodes[node]['T'],s,d,pi])
		print("results saved to {}".format(filename))
	return crowd, s_set, d_set, π_set

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Runs wisdom of the crowds. (Note: does not use the module, but had it defined inline because of issues with the current module version and Python 3.12+). \n\n Saves output to csv and optionally to a sullivan plot")
	utils.create_input_args(parser, ext=".grahpml")
	utils.create_output_args(parser, suffix="_woc.csv")	#TODO: This isn't actually used properly yet
	parser.add_argument("-c", "--communities", type=int, default=-1, help="Number of communites to consider (default: -1, which means each community). If set to n, will run the analysis for the top n communities. 0 means the full graph.")
	parser.add_argument("--plot", action='store_true', help="Generate sullivan plot (default: False).")
	parser.add_argument("--verbose", action='store_true', help="Enable verbose output for debugging and progress tracking")

	args = parser.parse_args()

	input_files = utils.parse_input_files_arg(args.input_file, ext=".graphml")
	output_files = utils.parse_output_files_arg(args.output, input_files)

	for input_file, output_file in zip(input_files, output_files):
		graph=nx.read_graphml(input_file)

		if args.communities == 0:
			wisdom_of_crowds(graph, plot=args.plot, filename=f"{os.path.splitext(input_file)[0]}", verbose=args.verbose)  # full graph
		
		elif args.communities > 0:
			for i in range(args.communities):
				wisdom_of_crowds(graph, i, plot=args.plot, verbose=args.verbose)
		else:
			# if communities is -1, we run the analysis for each community
			communities = set(nx.get_node_attributes(graph, 'T').values())
			for community in communities:
				wisdom_of_crowds(graph, int(community), plot=args.plot, filename=f"{os.path.splitext(input_file)[0]}", verbose=args.verbose)

