import importlib.util
from pathlib import Path
import matplotlib.pyplot as plt
import os
import csv

file_sets = {
	"ARG_scc-pruned_1000central": [
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\ARG_scc-pruned_0comm_1000central.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\ARG_scc-pruned_1comm_1000central.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\ARG_scc-pruned_2comm_1000central.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\ARG_scc-pruned_3comm_1000central.csv",
	],
	"ARG_scc-pruned_all": [
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\ARG_scc-pruned_0comm_all.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\ARG_scc-pruned_1comm_all.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\ARG_scc-pruned_2comm_all.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\ARG_scc-pruned_3comm_all.csv",
	],
	"ARG_wcc-pruned_1000central": [
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\ARG_wcc-pruned_0comm_1000central.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\ARG_wcc-pruned_1comm_1000central.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\ARG_wcc-pruned_2comm_1000central.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\ARG_wcc-pruned_3comm_1000central.csv",
	],
	"ARG_wcc-pruned_all": [
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\ARG_wcc-pruned_0comm_all.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\ARG_wcc-pruned_1comm_all.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\ARG_wcc-pruned_2comm_all.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\ARG_wcc-pruned_3comm_all.csv",
	],
	"AUS_scc-pruned_1000central": [
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\AUS_scc-pruned_0comm_1000central.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\AUS_scc-pruned_1comm_1000central.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\AUS_scc-pruned_2comm_1000central.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\AUS_scc-pruned_3comm_1000central.csv",
	],
	"AUS_scc-pruned_all": [
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\AUS_scc-pruned_0comm_all.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\AUS_scc-pruned_1comm_all.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\AUS_scc-pruned_2comm_all.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\AUS_scc-pruned_3comm_all.csv"
	],
	"AUS_wcc-pruned_1000central": [
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\AUS_wcc-pruned_0comm_1000central.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\AUS_wcc-pruned_1comm_1000central.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\AUS_wcc-pruned_2comm_1000central.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\AUS_wcc-pruned_3comm_1000central.csv"
	],
	"AUS_wcc-pruned_all": [
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\AUS_wcc-pruned_0comm_all.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\AUS_wcc-pruned_1comm_all.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\AUS_wcc-pruned_2comm_all.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\AUS_wcc-pruned_3comm_all.csv"
	],
	"USA_scc-pruned_1000central": [
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\USA_scc-pruned_0comm_1000central.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\USA_scc-pruned_1comm_1000central.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\USA_scc-pruned_2comm_1000central.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\USA_scc-pruned_3comm_1000central.csv",
	],
	"USA_scc-pruned_all": [
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\USA_scc-pruned_0comm_all.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\USA_scc-pruned_1comm_all.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\USA_scc-pruned_2comm_all.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\USA_scc-pruned_3comm_all.csv",
	],
	"USA_wcc-pruned_1000central": [
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\USA_wcc-pruned_0comm_1000central.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\USA_wcc-pruned_1comm_1000central.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\USA_wcc-pruned_2comm_1000central.csv",

	],
	"USA_wcc-pruned_all": [
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\USA_wcc-pruned_0comm_all.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\USA_wcc-pruned_1comm_all.csv",
		"C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\scores\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\USA_wcc-pruned_2comm_all.csv",
	],
}

columns_to_plot = [
	# "anger",
	# "anticipation",
	# "disgust",
	# "fear",
	# "joy",
	# "love",
	# "optimism",
	# "pessimism",
	# "sadness",
	# "surprise",
	# "trust"
]

community_info_files = {
	"ARG_scc-pruned": "C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\graph_building\\ARG\\community_colours.json",
	"ARG_wcc-pruned": "C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\graph_building\\ARG\\community_colours.json",
	"AUS_scc-pruned": "C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\graph_building\\AUS\\community_colours.json",
	"AUS_wcc-pruned": "C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\graph_building\\AUS\\community_colours.json",
	"USA_scc-pruned": "C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\graph_building\\USA\\community_colours.json",
	"USA_wcc-pruned": "C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\graph_building\\USA\\community_colours.json",
}


def _load_module(path: Path, module_name: str):
	"""Dynamically load a module from an arbitrary path (supports filenames with spaces)."""
	spec = importlib.util.spec_from_file_location(module_name, str(path))
	if spec is None or spec.loader is None:
		raise ImportError(f"Cannot load module {module_name} from {path}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)  # type: ignore[attr-defined]
	return module


viz_mod = _load_module(Path('10a - RoBERTa_viz.py'), 'viz_mod')
plot_func = getattr(viz_mod, 'plot')

for file_set_name, file_list in file_sets.items():
	if len(columns_to_plot) == 0:
		print(f"No columns specified to plot for file set '{file_set_name}'. Plotting as spectrum.")
		fig = plt.figure(figsize=(20, 10), facecolor='w')
		ax = fig.add_subplot(111)
		plt.title(f"RoBERTa Scores - {file_set_name} ", fontsize=16)
		plt.xlabel("Score Spectrum")
		plt.ylabel("Density")
		for file_name in file_list:
			print(f"Processing file: {file_name}")
			file_path = Path(file_name)
			if not file_path.is_file():
				print(f"File not found: {file_name}")
				continue

			print(f"Preparing to plot '{file_name}'")

			community_info_file = '_'.join(file_set_name.split('_')[:2])
			output_dir = Path("C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\visualizations\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\" + file_set_name + "\\")
			os.makedirs(output_dir, exist_ok=True)
			print("No columns specified, defaulting to all columns in CSV.")
			with open(file_name, "r", encoding="utf-8", errors="ignore") as f:
				reader = csv.DictReader(f)
				columns = reader.fieldnames[1:]
				print(f"Columns found: {columns}")

			plot_func(
				ax=ax,
				input_path=file_name,
				columns=columns,
				community_info=community_info_files[community_info_file],
				spectrum=True
			)
		plt.legend(title="Communities")
		plt.tight_layout()
		plt.savefig(output_dir.joinpath(f"{file_set_name}.png"), dpi=300)
		plt.close()
	else:
		for column in columns_to_plot:
			print(f"Preparing to plot column '{column}' for file set '{file_set_name}'")
			fig = plt.figure(figsize=(20, 10), facecolor='w')
			ax = fig.add_subplot(111)
			plt.title(f"RoBERTa Scores - {file_set_name} - {column}", fontsize=16)
			plt.xlabel("Emotion Score")
			plt.ylabel("Density")
			for file_name in file_list:
				print(f"Processing file: {file_name}")
				file_path = Path(file_name)
				if not file_path.is_file():
					print(f"File not found: {file_name}")
					continue
				
				print(f"Preparing to plot column '{column}' for file '{file_name}'")

				community_info_file = '_'.join(file_set_name.split('_')[:2])
				output_dir = Path("C:\\Users\\jabcm\\OneDrive\\PhD\\ISIS\\results\\RoBERTa\\outputs\\visualizations\\cardiffnlp\\twitter-roberta-base-sentiment-latest\\" + file_set_name + "\\")
				os.makedirs(output_dir, exist_ok=True)

				
				plot_func(
					ax=ax,
					input_path=file_name,
					columns=[column],
					community_info=community_info_files[community_info_file]
				)
			plt.legend(title="Communities")
			plt.tight_layout()
			plt.savefig(output_dir.joinpath(f"{file_set_name}_{column}.png"), dpi=300)
			plt.close()
