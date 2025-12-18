import argparse
import csv
import re
import utilities as util
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import seaborn as sns


def plot(ax, input_path, community_info, columns, spectrum=False):
	sentiment_weights = np.linspace(-1, 1, len(columns))
	
	comm_number = re.search(r'\d+', input_path).group(0)
	comm_name = util.get_community_label(comm_number, community_info)
	data = {}
	with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
		reader = csv.DictReader(f)
		for row in reader:
			if not spectrum:
				for col in columns:
					if col not in data:
						data[col] = []
					data[col].append(float(row[col]))
			else:

				# For each tweet's scores, calculate the continuum score
				if "spectrum" not in data:
					data["spectrum"] = []
				scores = [float(row[col]) for col in columns]
				score = np.dot(sentiment_weights, scores)
				data["spectrum"].append(score)

		if not spectrum:
			for col, values in data.items():
				sns.kdeplot(
					ax=ax,
					data=values,
					color=util.get_community_colour(comm_name, community_info),
					label=f"{comm_name}",
					linewidth=2
				)
		else:
			print(f"Plotting spectrum for community: {comm_name} with {len(data['spectrum'])} entries. Colour: {util.get_community_colour(comm_name, args.community_colours)}")
			sns.kdeplot(
				ax=ax,
				data=data["spectrum"],
				label=f"{comm_name}",
				linewidth=2,
				color=util.get_community_colour(comm_name, args.community_colours),
			)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Plots an existing RoBERTa output from a csv.")
	util.create_input_args(parser, ext=".csv", help="Input CSV file(s) containing RoBERTa output data.")
	util.create_output_args(parser, suffix="{plot_type}_{plot_community}.png")  # TODO: This isn't actually used properly yet
	parser.add_argument("-c", "--columns", type=str, required=False, help="Columns to plot for. Comma-separated list of column names from the CSV file. Defaults to all columns except the first")
	parser.add_argument("-s", "--spectrum", action="store_true", help="Whether to plot the values as a spectrum. If true, the leftmost column arg will be the low end of the spectrum, and the rightmost column arg the high end.")
	parser.add_argument("--community-colours",type=str,required=False,help="Path to a json file mapping community labels to colours, otherwise default colours will be used.")

	args = parser.parse_args()

	title = None

	input_paths = util.parse_input_files_arg(args.input_file, ext=".csv")
	output_paths = util.parse_output_files_arg(args.output, input_paths)

	fig = plt.figure(figsize=(20,10),facecolor='w')
	ax = fig.add_subplot(111)
	if args.spectrum:
		plt.title("Histogram of Sentiment Scores as a Continuum")
		plt.xlabel("Continuum Score (-1 = Strong Negative, 1 = Strong Positive)")
	else:
		if args.columns:
			plt.title(f"RoBERTa Sentiment Analysis KDE Plot for columns: {args.columns}")
		else:
			plt.title("RoBERTa Sentiment Analysis KDE Plot")
		plt.xlabel("Sentiment Score")
	plt.ylabel("Density")
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
			columns = columns,
			spectrum = args.spectrum
		)

	plt.legend(title="Communities")
	plt.tight_layout()
	plt.savefig(input_path.split('.csv')[0] + "_roBERta.png", dpi=300)
	plt.close()
