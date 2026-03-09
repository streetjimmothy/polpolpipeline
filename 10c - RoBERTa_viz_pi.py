import argparse
import csv
import re
import utilities as utils
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import seaborn as sns


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Plots an existing RoBERTa output from a csv.")
	utils.create_input_args(parser, ext=".csv", help="Input CSV file(s) containing RoBERTa output data.")
	utils.create_output_args(parser, suffix="{plot_type}_{plot_community}.png")  # TODO: This isn't actually used properly yet
	parser.add_argument("--threshold", "-t", type=float, default=0.0, help="Minimum sentiment score to include in the plot.")
	parser.add_argument("--community-colours","--community-info",type=str,required=False,help="Path to a json file mapping community labels to colours, otherwise default colours will be used.")

	args = parser.parse_args()

	title = None

	input_paths = utils.parse_input_files_arg(args.input_file, ext=".csv")
	output_paths = utils.parse_output_files_arg(args.output, input_paths)
	columns = [
			"anger",
			"anticipation",
			"disgust",
			"fear",
			"joy",
			"love",
			"optimism",
			"pessimism",
			"sadness",
			"surprise",
			"trust"
		]

	for col in columns:
		data = defaultdict(list)
		threshold = args.threshold

		for input_path in input_paths:
			filename = input_path.split('/')[-1]
			plot_name = filename[:3]
			comm_name = utils.get_community_label(
				re.search(r'\d', filename).group(0), 
				args.community_colours
			)
			with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
				reader = csv.DictReader(f)
				for row in reader:
					try:
						datum = float(row[col])
						if datum > threshold:
							data[comm_name].append(datum)
					except:
						continue


		labels = list(data.keys())
		sizes = [len(data[label]) for label in labels]
		colours = [utils.get_community_colour(label, args.community_colours) for label in labels]

		fig = plt.figure(figsize=(20, 10), facecolor='w')
		ax = fig.add_subplot(111)
		plt.title(f"Significant values for RoBERTa emotion: {col}")
		ax.pie(
			sizes, 
			labels=labels, 
			autopct='%1.1f%%',
			colors=colours
		)
			

		plt.legend(title="Communities")
		plt.tight_layout()
		plt.savefig(input_path.split('.csv')[0] + "_"+col+".png", dpi=300)
		plt.close()
