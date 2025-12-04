import time
import humanize
import psutil
import os
import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

def make_urgellplot_zoomed(nodes, colormap='cool', suptitle=None, cax=None, yscale='linear', filename=None, region=0.2):
	assert(len(nodes) > 0)

	# sort by pi, then d
	nodes = sorted(nodes, key=lambda n: (-n.S, -n.D))

	N = len(nodes)
	x = [i/(N-1) if N > 1 else 0.5 for i in range(N)]  # 0..1 inclusive

	cmap = plt.get_cmap(colormap)
	norm = Normalize(vmin=min(n.D for n in nodes)-1,vmax=max(n.D for n in nodes)+1)
	D_colours = [cmap(norm(n.D)) for n in nodes]

	
	# For bar centers across [0,1], use centers and width:
	π_bars_centers = [(i + 0.5)/N for i in range(N)]
	π_bars_width = 1.0 / N

	π_bars_height = [n.π for n in nodes]
	S_line = [n.S for n in nodes]
	# do the plot
	if cax == None:
		fig = plt.figure(figsize=(20,10),facecolor='w')
		ax = fig.add_subplot(111)
	else:
		ax = cax

	ax.bar(π_bars_centers,π_bars_height,width=π_bars_width,color=D_colours)
	ax.plot(x,S_line,c='k')

	ax.set_xticks([0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2])
	ax.set_xlim((0,region))

	ax.yaxis.tick_left()
	ax.yaxis.grid()

	# make the legend for D
	handles = []
	for d in set(n.D for n in nodes):
		handles.append(mpatches.Patch(color=cmap(norm(d)), label="D="+str(d)))
	ax.legend(handles=handles,loc='upper right')

	ax.set_yscale(yscale)

	if suptitle is not None:
		ax.set_title(suptitle)

	# Display inline in notebook; only save if a filename is provided
	ax.set_xlabel('Proportion')
	ax.set_ylabel('S/pi')
	if filename is not None:
		plt.savefig(filename, dpi=300, bbox_inches='tight')
	plt.show()
	return None

def make_urgellplot(nodes, colormap='cool', suptitle=None, cax=None, yscale='linear', filename=None):

	assert(len(nodes) > 0)

	# sort by pi, then d
	nodes = sorted(nodes, key=lambda n: (-n.S, -n.D))

	N = len(nodes)
	x = [i/(N-1) if N > 1 else 0.5 for i in range(N)]  # 0..1 inclusive

	cmap = plt.get_cmap(colormap)
	norm = Normalize(vmin=min(n.D for n in nodes)-1,vmax=max(n.D for n in nodes)+1)
	D_colours = [cmap(norm(n.D)) for n in nodes]

	
	# For bar centers across [0,1], use centers and width:
	π_bars_centers = [(i + 0.5)/N for i in range(N)]
	π_bars_width = 1.0 / N

	π_bars_height = [n.π for n in nodes]
	S_line = [n.S for n in nodes]
	# do the plot
	if cax == None:
		fig = plt.figure(figsize=(20,10),facecolor='w')
		ax = fig.add_subplot(111)
	else:
		ax = cax

	ax.bar(π_bars_centers,π_bars_height,width=π_bars_width,color=D_colours)
	ax.plot(x,S_line,c='k')

	ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
	ax.set_xlim((0,1))

	ax.yaxis.tick_left()
	ax.yaxis.grid()

	# make the legend for D
	handles = []
	for d in set(n.D for n in nodes):
		handles.append(mpatches.Patch(color=cmap(norm(d)), label="D="+str(d)))
	ax.legend(handles=handles,loc='upper right')

	ax.set_yscale(yscale)

	if suptitle is not None:
		ax.set_title(suptitle)

	# Display inline in notebook; only save if a filename is provided
	if cax == None:
		ax.set_xlabel('Proportion')
		ax.set_ylabel('S/pi')
		if filename is not None:
			plt.savefig(filename, dpi=300, bbox_inches='tight')
		plt.show()
		return None
	else:
		return None

def make_sullivanplot(nodes, colormap='viridis', suptitle=None, cax=None, yscale='linear', filename=None):

	assert(len(nodes) > 0)

	# sort by pi, then d
	nodes = sorted(nodes, key=lambda n: (n.π, n.D))

	N = len(nodes)
	x = [i/(N-1) if N > 1 else 0.5 for i in range(N)]  # 0..1 inclusive

	cmap = plt.get_cmap(colormap)
	norm = Normalize(vmin=min(n.D for n in nodes)-1,vmax=max(n.D for n in nodes)+1)
	D_colours = [cmap(norm(n.D)) for n in nodes]

	
	# For bar centers across [0,1], use centers and width:
	S_bars_centers = [(i + 0.5)/N for i in range(N)]
	S_bars_width = 1.0 / N

	S_bars_height = [n.S for n in nodes]
	π_line = [n.π for n in nodes]
	# do the plot
	if cax == None:
		fig = plt.figure(figsize=(12,6),facecolor='w')
		ax = fig.add_subplot(111)
	else:
		ax = cax

	ax.bar(S_bars_centers,S_bars_height,width=S_bars_width,color=D_colours)
	ax.plot(x,π_line,c='k')

	ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
	ax.set_xlim((0,1))

	ax.yaxis.tick_right()
	ax.yaxis.grid()

	# make the legend for D
	handles = []
	for d in set(n.D for n in nodes):
		handles.append(mpatches.Patch(color=cmap(norm(d)), label="D="+str(d)))
	ax.legend(handles=handles,loc='upper left')

	ax.set_yscale(yscale)

	if suptitle is not None:
		ax.set_title(suptitle)

	# Display inline in notebook; only save if a filename is provided
	if cax == None:
		ax.set_xlabel('Proportion')
		ax.set_ylabel('S/pi')
		if filename is not None:
			plt.savefig(filename, dpi=300, bbox_inches='tight')
		plt.show()
		return None
	else:
		return None

@dataclass(slots=True, frozen=True, order=True)
class NodeData:
	community: int
	S: int
	D: int
	π: int

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Plots an existing WoC output from a csv. \n Saves output to a sullivan plot")
	parser.add_argument("-i", "--input_file", required=True, help="Path to the input graphml file.")
	parser.add_argument("-o", "--output_file", required=False, help="Path to save the results (default: saves to input file + '.png' or input file + '_{community}.png').")
	parser.add_argument("-c", "--community", type=int, default=-1, help="Community to plot for (Special Values: -1 (default) - entire file is treated as one community; -2 - iterates over each community and plots each seperately). Will error if the input file does not have a community column.")
	parser.add_argument("-z", "--zoom", type=float, default=0.2, help="Zoom region for urgell plot (default 0.2 = 20%% of the x axis). Only applies to urgell plots.")
	parser.add_argument("-p", "--plot_type", type=str, default="urgell", choices=["urgell", "sullivan"], help="Type of plot to generate (default: urgell).")
	parser.add_argument("-m", "--merge_plots", action='store_true', help="If set, merges all community plots into a single plot (only applicable when plotting all communities).")
	parser.add_argument("-d", "--max-degree", type=int, default=9, help="Maximum degree to consider when plotting (default: 9). Nodes with degree higher than this will be treated as having this degree.")

	args = parser.parse_args()
	
	title = None


	with open(args.input_file, 'r') as infile:
		reader = csv.DictReader(infile)
		# make one plot per community
		comm_dict = defaultdict(list)  # maps community -> (list of pi, list of D, list of S)
		if args.community == -1:
			for row in reader:
				_S = int(row['S'])
				_D = min(int(row['D']), args.max_degree) 
				_π = _S * _D
				comm_dict[0].append(
					NodeData(
						community = 0,
						S = _S,
						D = _D,
						π = _π
					)
				)
		else:
			for row in reader:
				comm = int(float(row['community']))
				if comm < 4 and (args.community == -2 or comm == args.community):
					_S = int(row['S'])
					_D = min(int(row['D']), args.max_degree) 
					_π = _S * _D
					comm_dict[comm].append(
						NodeData(
							community = int(float(row['community'])),
							S = _S,
							D = _D,
							π = _π
						)
					)
		num_comms = len(comm_dict)
		comm_dict = dict(sorted(comm_dict.items()))  # sort by community number
		print(f"Loaded {len(comm_dict)} communities from {args.input_file}")

		merge_axes = None
		if args.merge_plots and args.community == -2:
			fig = plt.figure(figsize=(20,10),facecolor='w')
			if num_comms == 1:
				merge_axes = [fig.add_subplot(111)]
			elif num_comms == 2:
				merge_axes = [fig.add_subplot(121), fig.add_subplot(122)]
			elif num_comms == 3:
				merge_axes = [fig.add_subplot(311), fig.add_subplot(312), fig.add_subplot(313)]
			elif num_comms == 4:
				merge_axes = [fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)]
			elif num_comms == 5:
				merge_axes = [fig.add_subplot(511), fig.add_subplot(512), fig.add_subplot(513), fig.add_subplot(514), fig.add_subplot(515)]
			elif num_comms == 6:
				merge_axes = [fig.add_subplot(231), fig.add_subplot(232), fig.add_subplot(233), fig.add_subplot(234), fig.add_subplot(235), fig.add_subplot(236)]


		for comm in comm_dict:
			if args.community == -1:
				suptitle = "Entire Graph (n="+str(len(comm_dict[comm]))+")"
				output_file = f'{os.path.splitext(args.input_file)[0]}-urgell_plot_zoomed.png'
				make_urgellplot_zoomed(comm_dict[comm], suptitle=suptitle, filename=output_file, region=args.zoom, cax=None)
				print(f"Saved entire graph plot to {output_file}")
			elif args.community == comm or args.community == -2:
				suptitle = "Community "+str(comm)+" (n="+str(len(comm_dict[comm]))+")"
				if args.merge_plots:
					output_file = f'{os.path.splitext(args.input_file)[0]}-all_communities_plot.png'
				else:
					output_file = f'{os.path.splitext(args.input_file)[0]}_{comm}-urgell_plot_zoomed.png'
				make_urgellplot_zoomed(comm_dict[comm], suptitle=suptitle, filename=output_file, region=args.zoom, cax=merge_axes[comm] if merge_axes else None)
				print(f"Saved community {comm} plot to {output_file}")


