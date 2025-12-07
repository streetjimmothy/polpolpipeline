import os
import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import math
import utilities as util

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

	# Set 6 evenly spaced ticks from 0 to region (inclusive)
	num_ticks = 8
	xticks = [region * i / (num_ticks - 1) for i in range(num_ticks)]
	ax.set_xticks(xticks)
	ax.set_xticklabels([f"{tick:.3g}" for tick in xticks])
	ax.set_xlim((0, region))

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
	util.create_input_args(parser, ext=".csv")
	util.create_output_args(parser, suffix="{plot_type}_{plot_community}.png")	#TODO: This isn't actually used properly yet
	parser.add_argument("-c", "--community", type=int, default=-1, help="Community to plot for (Special Values: -1 (default) - entire file is treated as one community; -2 - iterates over each community and plots each seperately). Will error if the input file does not have a community column.")
	parser.add_argument("-z", "--zoom", type=float, default=0.2, help="Zoom region for urgell plot (default 0.2 = 20% of the x axis). Only applies to urgell plots.")
	parser.add_argument("-p", "--plot_type", type=str, default="urgell", choices=["urgell", "sullivan"], help="Type of plot to generate (default: urgell).")
	parser.add_argument("-m", "--merge_plots", action='store_true', help="If set, merges all community plots into a single plot (only applicable when plotting all communities).")
	parser.add_argument("-d", "--max-degree", type=int, default=9, help="Maximum degree to consider when plotting (default: 9). Nodes with degree higher than this will be treated as having this degree.")
	parser.add_argument("--comm_min", type=float, default=0.0, help="Threshold for communities to plot. If n<1, plot all communites larger than n% of the total graph size. If n>1 plot n largest communities (default: 0.0, plot all).")

	args = parser.parse_args()
	
	title = None

	input_paths = util.parse_input_files_arg(args.input_file, ext=".csv")
	output_paths = util.parse_output_files_arg(args.output, input_paths)

	for input_path, output_path in zip(input_paths, output_paths):
		with open(input_path, 'r') as infile:
			reader = csv.DictReader(infile)
			# make one plot per community
			comm_dict = defaultdict(list)  # maps community -> (list of pi, list of D, list of S)
			num_nodes = 0
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
					num_nodes += 1
			else:
				for row in reader:
					comm = int(float(row['Community']))
					if comm < 4 and (args.community == -2 or comm == args.community):
						_S = int(row['S'])
						_D = min(int(row['D']), args.max_degree) 
						_π = _S * _D
						comm_dict[comm].append(
							NodeData(
								community = int(float(row['Community'])),
								S = _S,
								D = _D,
								π = _π
							)
						)
					num_nodes += 1
			# Filter communities by minimum size fraction if requested
			if args.comm_min > 1:
				# Keep only the n largest communities
				sorted_comms = sorted(comm_dict.items(), key=lambda item: len(item[1]), reverse=True)
				comm_dict = dict(sorted_comms[:int(args.comm_min)])
			else:
				comm_dict = dict(sorted(comm_dict.items()))  # sort by community number
				min_frac = args.comm_min
				if min_frac > 0:
					comm_dict = {k: v for k, v in comm_dict.items() if len(v) >= min_frac * num_nodes }
				num_comms = len(comm_dict)
				print(f"Loaded {num_comms} communities from {input_path} (threshold: {min_frac})")

			if num_comms == 0:
				print("No communities to plot after applying filters.")
				continue

			merge_axes = None
			if args.merge_plots and args.community == -2:
				# Compute grid shape as square as possible
				n = num_comms
				ncols = math.ceil(math.sqrt(n))
				nrows = math.ceil(n / ncols)
				fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 4*nrows), facecolor='w')
				axes = axes.flatten() if n > 1 else [axes]
				# Remove unused axes and center last row
				if n < nrows * ncols:
					# Hide unused axes
					for ax in axes[n:]:
						ax.set_visible(False)
					# Center last row
					last_row_start = (nrows - 1) * ncols
					last_row_count = n - last_row_start
					if last_row_count < ncols:
						offset = (ncols - last_row_count) // 2
						for i in range(ncols):
							idx = last_row_start + i
							if i < offset or i >= offset + last_row_count:
								axes[idx].set_visible(False)
				merge_axes = axes[:n]
				plt.suptitle(f"All Communities from {os.path.basename(input_path)}", fontsize=16)
				plt.tight_layout(pad=3)
				#fig.tight_layout()
				#fig.subplots_adjust(wspace=0.3, hspace=0.4)

			for idx, (comm, value) in enumerate(comm_dict.items()):
				if args.community == -1:
					suptitle = "Entire Graph (n="+str(len(value))+")"
					output_file = f'{output_path}-urgell_plot_zoomed.png'
					make_urgellplot_zoomed(value, suptitle=suptitle, filename=output_file, region=args.zoom, cax=None)
					print(f"Saved entire graph plot to {output_file}")
				elif args.community == comm or args.community == -2:
					print("Plotting Community " + str(comm) + " (n=" + str(len(value)) + ")")
					suptitle = "Community "+str(comm)+" (n="+str(len(value))+")"
					if args.merge_plots:
						output_file = f'{output_path}-all_communities_plot.png'
					else:
						output_file = f'{output_path}_{comm}-urgell_plot_zoomed.png'
					make_urgellplot_zoomed(value, suptitle=suptitle, filename=output_file, region=args.zoom, cax=merge_axes[idx] if len(merge_axes) else None)
					print(f"Saved community {comm} plot to {output_file}")


