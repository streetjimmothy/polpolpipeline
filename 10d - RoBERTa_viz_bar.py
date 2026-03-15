import argparse
import csv
import re
import utilities as utils
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches

def plot(ax, input_path, community_info, columns):
	comm_number = re.search(r'\d+', input_path).group(0)
	comm_name = utils.get_community_label(comm_number, community_info)
	data = {}
	with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
		reader = csv.DictReader(f)
		for row in reader:
			for col in columns:
				if col not in data:
					data[col] = []
				data[col].append(float(row[col]))

	color = utils.get_community_colour(comm_name, community_info)

	if not hasattr(ax, '_pending_plots'):
		ax._pending_plots = []
	ax._pending_plots.append((data, color, comm_name))


def finalise_plot(ax, columns):
	if not hasattr(ax, '_pending_plots') or not ax._pending_plots:
		return []

	n_comms = len(ax._pending_plots)
	n_cols = len(columns)
	box_width = 1 / n_comms
	group_gap = 1.5
	legend_handles = []

	for comm_idx, (data, color, comm_name) in enumerate(ax._pending_plots):
		offset = (comm_idx - (n_comms - 1) / 2) * box_width
		positions = [i * group_gap + offset for i in range(n_cols)]

		bp = ax.boxplot(
			[data[col] for col in columns],
			positions=positions,
			patch_artist=True,
			widths=box_width * 0.5,
			showfliers=False,
			medianprops=dict(color='black', linewidth=2)
		)

		for patch in bp['boxes']:
			patch.set_facecolor(color)
			patch.set_alpha(0.7)

		legend_handles.append(mpatches.Patch(facecolor=color, alpha=0.7, label=comm_name))

	ax.legend(title="Communities", handles=legend_handles)

	# Centre x-tick labels on each column group
	group_centres = [i * group_gap for i in range(n_cols)]
	ax.set_xticks(group_centres)
	ax.set_xticklabels(columns, rotation=45, ha='right')



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Plots an existing RoBERTa output from a csv.")
	utils.create_input_args(parser, ext=".csv", help="Input CSV file(s) containing RoBERTa output data.")
	utils.create_output_args(parser, suffix="{plot_type}_{plot_community}.png")  # TODO: This isn't actually used properly yet
	parser.add_argument("-c", "--columns", type=str, required=False, help="Columns to plot for. Comma-separated list of column names from the CSV file. Defaults to all columns except the first")
	parser.add_argument("--community-colours",type=str,required=False,help="Path to a json file mapping community labels to colours, otherwise default colours will be used.")

	args = parser.parse_args()

	title = None

	input_paths = utils.parse_input_files_arg(args.input_file, ext=".csv")
	output_paths = utils.parse_output_files_arg(args.output, input_paths)

	fig = plt.figure(figsize=(20,10),facecolor='w')
	ax = fig.add_subplot(111)
	if args.columns:
		pass#plt.title(f"RoBERTa Sentiment Analysis KDE Plot for columns: {args.columns}")
	else:
		pass#plt.title("RoBERTa Sentiment Analysis KDE Plot")
	#plt.xlabel("Sentiment Polarity")
	#plt.ylabel("Density")
	columns = None
	if not args.columns:
		print("No columns specified, defaulting to all columns in CSV.")
		with open(input_paths[0], "r", encoding="utf-8", errors="ignore") as f:
			reader = csv.DictReader(f)
			columns = reader.fieldnames[1:]
			print(f"Columns found: {columns}")
	else:
		columns = [col.strip() for col in args.columns.split(",")]

	for input_path in input_paths:
		plot(
			ax=ax, 
			input_path=input_path, 
			community_info=args.community_colours if args.community_colours else None,
			columns = columns
		)

	finalise_plot(ax, columns)

	plt.tight_layout()
	print(f"Saving plot to {args.output if args.output else input_path.split('.csv')[0] + '_roBERta.png'}")
	if args.output:
		plt.savefig(args.output, dpi=300)
	else:
		plt.savefig(input_path.split('.csv')[0] + "_roBERta.png", dpi=300)
	plt.close()
