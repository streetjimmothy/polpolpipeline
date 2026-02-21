import os
import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import math
import utilities as utils


def make_ccdf(nodes, yscale='linear', colour="k", suptitle=None, cax=None, filename=None,):
	assert(len(nodes) > 0)

	N = len(nodes)
	counts = Counter(n.π for n in nodes)
	π_values = list(range(0, 101))
	ccdf = []
	running = 0
	for π in range(100, -1, -1):
		running += counts.get(π, 0)
		ccdf.append(running / N)
	ccdf.reverse()

	# do the plot
	if cax == None:
		fig = plt.figure(figsize=(20,10),facecolor='w')
		ax = fig.add_subplot(111)
	else:
		ax = cax
		
	ax.step(ccdf, π_values, where='post', c=colour)
	ax.set_xlim((0, 1))
	ax.set_ylim((0, 100))

	ax.yaxis.tick_left()
	ax.yaxis.grid()

	ax.set_yscale(yscale)

	if suptitle is not None:
		ax.set_title(suptitle)

	# Display inline in notebook; only save if a filename is provided
	ax.set_xlabel('CCDF P(π ≥ s)')
	ax.set_ylabel('π value')
	if filename is not None:
		plt.savefig(filename, dpi=300, bbox_inches='tight')
	else:
		plt.show()
	return None


def make_ccdf_cumulative_integral(nodes, yscale='linear', colour="k", suptitle=None, cax=None, filename=None,):
	assert(len(nodes) > 0)

	N = len(nodes)
	counts = Counter(n.π for n in nodes)
	π_values = list(range(0, 101))
	ccdf = []
	running = 0
	for π in range(100, -1, -1):
		running += counts.get(π, 0)
		ccdf.append(running / N)
	ccdf.reverse()

	ccdf_cumint = []
	ccdf_cumint = [0.0] * len(ccdf)
	running_integral = 0.0
	for i in range(len(ccdf) - 1, -1, -1):
		running_integral += ccdf[i]
		ccdf_cumint[i] = running_integral

	if cax == None:
		fig = plt.figure(figsize=(20,10),facecolor='w')
		ax = fig.add_subplot(111)
	else:
		ax = cax

	ax.step(ccdf_cumint, π_values, where='post', c=colour)
	ax.set_xlim((0, max(ccdf_cumint) if len(ccdf_cumint) > 0 else 1))
	ax.set_ylim((0, 100))

	ax.yaxis.tick_left()
	ax.yaxis.grid()

	ax.set_yscale(yscale)

	if suptitle is not None:
		ax.set_title(suptitle)

	ax.set_xlabel('Cumulative Integral of CCDF')
	ax.set_ylabel('π value')
	if filename is not None:
		plt.savefig(filename, dpi=300, bbox_inches='tight')
	else:
		plt.show()
	return None


def compute_ccdf_and_area(values, vmin=0, vmax=100, normalize=False):
    """
    Compute CCDF from a sequence of integer values and return (x, ccdf, area).

    - values: iterable of ints (or int-castable)
    - vmin/vmax: support bounds (inclusive)
    - normalize=True -> divide area by number of x-points
    """
    values = [min(max(int(v), vmin), vmax) for v in values]
    if not values:
        raise ValueError("values must not be empty")

    N = len(values)
    counts = Counter(values)
    x = list(range(vmin, vmax + 1))

    ccdf = []
    running = 0
    for s in range(vmax, vmin - 1, -1):
        running += counts.get(s, 0)
        ccdf.append(running / N)
    ccdf.reverse()  # align with x ascending

    # Area under CCDF with unit spacing on integer support
    area = sum(ccdf)
    if normalize:
        area /= len(ccdf)

    print(f"CCDF area: {area:.6f}")
    return x, ccdf, area

def make_kde_survival(nodes, yscale='linear', colour="k", suptitle=None, cax=None, filename=None, bandwidth=None):
	assert(len(nodes) > 0)

	n_support = 100
	π_values = list(range(0, n_support + 1))
	samples = [min(max(int(n.π), 0), n_support) for n in nodes]
	N = len(samples)

	if bandwidth is None:
		if N > 1:
			mean_π = sum(samples) / N
			var_π = sum((π - mean_π) ** 2 for π in samples) / (N - 1)
			std_π = math.sqrt(max(var_π, 0.0))
			bandwidth = max(1.0, 1.06 * std_π * (N ** (-1 / 5)))
		else:
			bandwidth = 1.0

	counts = Counter(samples)
	sqrt_2pi = math.sqrt(2 * math.pi)
	pmf = []
	for x in π_values:
		density = 0.0
		for s, c in counts.items():
			weight = c / N
			z_main = (x - s) / bandwidth
			z_low_reflect = (x + s) / bandwidth
			z_high_reflect = (x - (2 * n_support - s)) / bandwidth
			density += weight * (
				math.exp(-0.5 * z_main * z_main)
				+ math.exp(-0.5 * z_low_reflect * z_low_reflect)
				+ math.exp(-0.5 * z_high_reflect * z_high_reflect)
			)
		pmf.append(density / (bandwidth * sqrt_2pi))

	total = sum(pmf)
	if total > 0:
		pmf = [v / total for v in pmf]

	sf = [0.0] * (n_support + 1)
	running = 0.0
	for π in range(n_support, -1, -1):
		running += pmf[π]
		sf[π] = running

	if cax == None:
		fig = plt.figure(figsize=(20,10),facecolor='w')
		ax = fig.add_subplot(111)
	else:
		ax = cax

	ax.plot(sf, π_values, where='post', c=colour)
	ax.set_xlim((0, 1))
	ax.set_ylim((0, n_support))

	ax.yaxis.tick_left()
	ax.yaxis.grid()

	ax.set_yscale(yscale)

	if suptitle is not None:
		ax.set_title(suptitle)

	ax.set_xlabel('KDE SF P(π ≥ s)')
	ax.set_ylabel('π value')
	if filename is not None:
		plt.savefig(filename, dpi=300, bbox_inches='tight')
	else:
		plt.show()
	return None


def make_betabinomial_survival(nodes, yscale='linear', colour="k", suptitle=None, cax=None, filename=None,):
	assert(len(nodes) > 0)

	n_support = 100
	π_values = list(range(0, n_support + 1))
	samples = [min(max(int(n.π), 0), n_support) for n in nodes]
	N = len(samples)
	counts = Counter(samples)
	empirical_pmf = [counts.get(π, 0) / N for π in π_values]
	mean_π = sum(samples) / N

	if N > 1:
		var_π = sum((π - mean_π) ** 2 for π in samples) / (N - 1)
	else:
		var_π = 0.0

	beta_binomial_pmf = empirical_pmf[:]
	if mean_π <= 0:
		beta_binomial_pmf = [1.0] + [0.0] * n_support
	elif mean_π >= n_support:
		beta_binomial_pmf = [0.0] * n_support + [1.0]
	else:
		p = mean_π / n_support
		base_var = n_support * p * (1 - p)
		if base_var > 0:
			inflation = var_π / base_var
			rho = (inflation - 1.0) / (n_support - 1)
			rho = min(max(rho, 1e-6), 0.999999)
			t = (1.0 / rho) - 1.0
			alpha = max(p * t, 1e-6)
			beta = max((1 - p) * t, 1e-6)

			pmf = []
			log_beta_norm = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
			for k in π_values:
				log_choose = math.lgamma(n_support + 1) - math.lgamma(k + 1) - math.lgamma(n_support - k + 1)
				log_beta_num = math.lgamma(k + alpha) + math.lgamma(n_support - k + beta) - math.lgamma(n_support + alpha + beta)
				pmf.append(math.exp(log_choose + log_beta_num - log_beta_norm))

			total = sum(pmf)
			if total > 0:
				beta_binomial_pmf = [v / total for v in pmf]

	smooth_weight = 1
	blended_pmf = [
		smooth_weight * model + (1.0 - smooth_weight) * empirical
		for model, empirical in zip(beta_binomial_pmf, empirical_pmf)
	]
	p_total = sum(blended_pmf)
	if p_total > 0:
		blended_pmf = [v / p_total for v in blended_pmf]

	sf = [0.0] * (n_support + 1)
	running = 0.0
	for π in range(n_support, -1, -1):
		running += blended_pmf[π]
		sf[π] = running

	if cax == None:
		fig = plt.figure(figsize=(20,10),facecolor='w')
		ax = fig.add_subplot(111)
	else:
		ax = cax

	ax.step(sf, π_values, where='post', c=colour)
	ax.set_xlim((0, 1))
	ax.set_ylim((0, n_support))

	ax.yaxis.tick_left()
	ax.yaxis.grid()

	ax.set_yscale(yscale)

	if suptitle is not None:
		ax.set_title(suptitle)

	ax.set_xlabel('Beta-Binomial SF P(S ≥ s)')
	ax.set_ylabel('S value')
	if filename is not None:
		plt.savefig(filename, dpi=300, bbox_inches='tight')
	else:
		plt.show()
	return None

@dataclass(slots=True, frozen=True, order=True)
class NodeData:
	community: int
	S: int
	D: int
	π: int


def community_min_param(value):
	if '.' in value:
		fvalue = float(value)
		if fvalue < 0.0 or fvalue >= 1.0:
			raise argparse.ArgumentTypeError(f"Invalid community min fraction: {value}. Must be in range [0.0, 1.0).")
		return fvalue
	else:
		ivalue = int(value)
		if ivalue < 1:
			raise argparse.ArgumentTypeError(f"Invalid community min count: {value}. Must be an integer >= 1.")
		return ivalue

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Plots an existing WoC output from a csv. \n Saves output to a sullivan plot")
	utils.create_input_args(parser, ext=".csv")
	utils.create_output_args(parser, suffix="{plot_type}_{plot_community}.png")	#TODO: This isn't actually used properly yet
	parser.add_argument("-c", "--community", type=int, default=-1, help="Community to plot for (Special Values: -1 (default) - entire file is treated as one community; -2 - iterates over each community and plots each seperately). Will error if the input file does not have a community column.")
	parser.add_argument("-z", "--zoom", type=float, default=0.2, help="Zoom region for urgell plot (default 0.2 = 20%% of the x axis). Only applies to urgell plots.")
	parser.add_argument("-p", "--plot-type", type=str, default="urgell", choices=["urgell", "sullivan"], help="Type of plot to generate (default: urgell).")
	parser.add_argument("--merge-plots", action='store_true', help="If set, merges all community plots into a single output side-by-side (only applicable when plotting all communities). Incompatible with --overlay.")
	parser.add_argument("--overlay", action='store_true', help="If set, overlays all community plots on the same axes (only applicable when plotting all communities). Incompatible with --merge-plots.")
	parser.add_argument("--merge_files", action='store_true', help="If set, merges all input files into a single plot (only applicable when plotting all communities). Requires --merge-plots or --overlay to be set. With --merge-plots, prints all communities from each file on the same output. With --overlay, overlays all communities from all files on the same axes (only recommended for small number of communities or with --community).")
	parser.add_argument("-d", "--max-degree", type=int, default=9, help="Maximum degree to consider when plotting (default: 9). Nodes with degree higher than this will be treated as having this degree.")
	parser.add_argument("--comm-min", type=community_min_param, default=0.0, help="Threshold for communities to plot. If n<1, plot all communites larger than n%% of the total graph size. If n>1 plot n largest communities (default: 0.0, plot all).")
	parser.add_argument("--community-info",type=str,required=True,help="Path to a json file with community name and colour info")

	args = parser.parse_args()

	if args.overlay and args.merge_plots:
		print("Error: --overlay and --merge_plots cannot be used together.")
	
	title = None

	input_paths = utils.parse_input_files_arg(args.input_file, ext=".csv")
	output_paths = utils.parse_output_files_arg(args.output, input_paths)

	data = {}
	for input_path in input_paths:
		filename = os.path.splitext(os.path.basename(input_path))[0]
		data[filename] = []
		with open(input_path, 'r') as infile:
			reader = csv.DictReader(infile)
			comm_dict = defaultdict(list)  # maps community -> (list of pi, list of D, list of S)
			num_nodes = 0
			for row in reader:
				comm = utils.get_community_label(int(float(row['Community'])), args.community_info)
				_S = int(row['S'])
				_D = min(int(row['D']), args.max_degree)
				_π = _S * _D
				comm_dict[comm].append(
					NodeData(
						community=comm,
						S=_S,
						D=_D,
						π=_π
					)
				)
				num_nodes += 1
			data[filename] = comm_dict


	for filename, comm_dict in data.items():
		if args.comm_min > 1:
			# Keep only the n largest communities
			sorted_comms = sorted(comm_dict.items(), key=lambda item: len(item[1]), reverse=True)
			comm_dict = dict(sorted_comms[:int(args.comm_min)])
			num_comms = len(comm_dict)
			print(f"Loaded {num_comms} communities from {input_path} (top {int(args.comm_min)} largest)")
		else:
			comm_dict = dict(sorted(comm_dict.items()))  # sort by community number
			min_frac = args.comm_min
			if min_frac > 0:
				comm_dict = {k: v for k, v in comm_dict.items() if len(v) >= min_frac * num_nodes }
			num_comms = len(comm_dict)
			print(f"Loaded {num_comms} communities from {input_path} (threshold: {min_frac})")
		data[filename] = comm_dict

	num_comms = 0
	if args.community != -1:
		all_gone = True
		for filename, comm_dict in data.items():
			print(f"Communities in {filename}:")
			if len(comm_dict) == 0:
				print("No communities passed the filtering criteria for file {}".format(filename))
				continue
			else:
				all_gone = False
			for comm, nodes in comm_dict.items():
				print(f"  Community {comm}: {len(nodes)} nodes")
				num_comms += 1
		if all_gone:
			print("No communities passed the filtering criteria for any input file. Exiting.")
			exit(0)
	else:
		for filename, comm_dict in data.items():
			merged = []
			for comm in comm_dict.values():
				for node in comm:
					merged.append(node)
			data[filename] = {"all": merged}

	merge_axes = None
	if args.merge_plots:
		# Compute grid shape as square as possible
		n = num_comms
		ncols = math.ceil(math.sqrt(n))
		nrows = math.ceil(n / ncols)
		fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4 * nrows), facecolor='w')
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

	if args.overlay:
		if args.output is None and len(input_paths) != 1:
			#TODO: WTF?
			print("Error: --overlay requires an output file to be specified if more than one input is provied.")
			exit(1)
		else:
			output_file = f'{output_paths[0]}_overlay_plot.png'
		overlay_data = {}
		output_filebase = os.path.splitext(os.path.basename(output_paths[0]))[0]
		overlay_data[output_filebase] = {}
		for filename, comm_dict in data.items():
			overlay_data[output_filebase].update(comm_dict)
		data = overlay_data
		overlay_axes = plt.figure(figsize=(12, 12), facecolor='w').add_subplot(111)
		suptitle = f"All communites for {filename}"
	
	for idx, output_path in enumerate(output_paths):
		if not args.merge_plots and not args.overlay:
			if os.path.exists(output_path):
				print(f"Warning: Output file {output_path} already exists and will be overwritten.")
			filename, comm_dict = list(data.items())[idx]
			if args.community == -1:
				all_nodes = comm_dict.get("all", [])
				suptitle = f"All communites merged for {filename} (n={str(len(all_nodes))})"
				output_file = f'{output_path}-urgell_plot_zoomed.png'
				make_ccdf(all_nodes, suptitle=suptitle, filename=output_file, cax=None)
				print(f"Saved entire graph plot to {output_file}")
		if args.overlay:
			#output_file = f'{output_path}_overlay_plot.png'
			#output_file = f'{output_path}_overlay_betabinomial_plot.png'
			#output_file = f'{output_path}_overlay_kde_plot.png'
			output_file = f'{output_path}_overlay_CAUC_plot.png'
			legend_patches = {}
			for comm, nodes in comm_dict.items():
				if args.community == comm or args.community == -2:
					color = utils.get_community_colour(comm, args.community_info)
					label = utils.get_community_label(comm, args.community_info)
					if label not in legend_patches:
						legend_patches[label] = mpatches.Patch(color=color, label=label)
					#make_ccdf(value, suptitle=suptitle, filename=output_file, cax=overlay_axes, colour=color)
					#make_betabinomial_survival(value, suptitle=suptitle, filename=bi_output_file, cax=overlay_axes, colour=color)
					#make_kde_survival(value, suptitle=suptitle, filename=output_file, cax=overlay_axes, colour=color)
					pi_values = [n.π for n in nodes]
					print(f"Community {comm} - mean π: {sum(pi_values)/len(pi_values):.2f}")
					x, ccdf, area = compute_ccdf_and_area(pi_values, vmin=0, vmax=100)
					make_ccdf_cumulative_integral(nodes, suptitle=suptitle, filename=output_file, cax=overlay_axes, colour=color)
			overlay_axes.legend(handles=list(legend_patches.values()), loc="upper right")
			overlay_axes.figure.savefig(output_file, dpi=300, bbox_inches="tight")
		if args.merge_plots:
			for comm, nodes in comm_dict.items():
				if args.community == comm or args.community == -2:
					print(f"Plotting Community {str(comm)}: {utils.get_community_label(comm, args.community_info)} (n={str(len(nodes))})")
					suptitle = f"{utils.get_community_label(comm, args.community_info)} (n={str(len(nodes))})"
					if args.merge_plots:
						output_file = f'{output_path}-all_communities_plot.png'
						make_ccdf(nodes, suptitle=suptitle, filename=output_file, cax=merge_axes[idx] if len(merge_axes) else None)
					else:
						#TODO This doesn't make sense here
						output_file = f'{output_path}_{comm}-urgell_plot_zoomed.png'
						make_ccdf(nodes, suptitle=suptitle, filename=output_file, cax=None)
					print(f"Saved community {comm} plot to {output_file}")


