from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import torch

import argparse
try:
	from tqdm import tqdm
except ImportError:
	tqdm = None


def run_classification(model_name, labels, test_text, verbose=False):
	# Load the model and tokenizer
	model = AutoModelForSequenceClassification .from_pretrained(model_name)
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model.to(device).eval()
	tweet_scores = {}

	# distribution by category
	distribution = {label: [] for label in labels.values()}

	# count/proportion in each category by threshold
	threshold = 0.25
	counts = {label: 0 for label in labels.values()}

	batch_size = 256  # You can adjust this based on your hardware

	iterator = range(0, len(test_text), batch_size)
	if verbose and tqdm:
		iterator = tqdm(iterator, desc="Classifying")
	for start in iterator:
		batch_lines = test_text[start:start + batch_size]
		encoded_input = tokenizer(
			batch_lines,
			return_tensors='pt',
			padding=True,
			truncation=True,
			max_length=512
		)
		encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
		with torch.inference_mode():
			output = model(**encoded_input)
			logits = output.logits
			batch_scores = torch.softmax(logits, dim=1).cpu().numpy()  # shape: (batch_size, num_classes)
		for idx, scores in enumerate(batch_scores):
			for i, label in labels.items():
				distribution[labels[i]].append(scores[i])
				if scores[i] > threshold:
					counts[labels[i]] += 1
			tweet_scores[batch_lines[idx]] = scores
	return tweet_scores, distribution, counts


models_labels = {
	"cardiffnlp/twitter-roberta-large-emotion-latest":
		{
			0: "anger",
			1: "anticipation",
			2: "disgust",
			3: "fear",
			4: "joy",
			5: "love",
			6: "optimism",
			7: "pessimism",
			8: "sadness",
			9: "surprise",
			10: "trust"
		},
	"cardiffnlp/twitter-roberta-base-sentiment-latest":
		{
			0: "negative",
			1: "neutral",
			2: "positive"
		},
	"cardiffnlp/twitter-roberta-large-topic-sentiment-latest":
		{
			0: "strongly negative",
			1: "negative",
			2: "negative or neutral",
			3: "positive",
			4: "strongly positive"
		},
	"cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual":  # primarily for ARG, but applied uniformly
		{
			0: "negative",
			1: "neutral",
			2: "positive"
		},
	# "cardiffnlp/twitter-xlm-roberta-base-sentiment": #has issues with the tokeniser
	# 	{
	# 		0: "negative",
	# 		1: "neutral",
	# 		2: "positive"
	# 	},
	"dccuchile/bert-base-spanish-wwm-cased":
		{
			0: "negative",
			1: "positive"
		},
	"VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis":
		{
			0: "negative",
			1: "positive"
		}
}


def save_images(distribution, counts, all_scores, labels, model_name, out_dir, filename):
	directory = f"{out_dir}/images/{model_name}/"
	if not os.path.exists(directory):
		os.makedirs(directory)

	print(f"Saving images for {model_name} - {filename}")
	# Plot distribution of scores
	print(f"Plotting distribution for {model_name} - {filename}")
	plt.figure(figsize=(10, 6))
	for label, scores in distribution.items():
		sns.kdeplot(scores, label=label, fill=True)
	plt.title(f"Score Distribution for {model_name}")
	plt.xlabel("Scores")
	plt.ylabel("Density")
	plt.legend()
	plt.savefig(os.path.join(directory, f"{filename}_distribution.png"))
	plt.close()

	# Plot counts
	print(f"Plotting counts for {model_name} - {filename}")
	plt.figure(figsize=(10, 6))
	sns.barplot(x=list(counts.keys()), y=list(counts.values()))
	plt.title(f"Counts of Categories for {model_name}")
	plt.xlabel("Categories")
	plt.ylabel("Counts")
	plt.xticks(rotation=45)
	plt.savefig(os.path.join(directory, f"{filename}_counts.png"))
	plt.close()

	scores_to_plot = all_scores.flatten()
	title = "Distribution for all classes"
	plt.figure(figsize=(8, 4))
	sns.histplot(scores_to_plot, kde=True, bins=50)
	plt.title(title)
	plt.xlabel("Score")
	plt.ylabel("Frequency")
	plt.savefig(os.path.join(directory, f"{filename}_distribution-allclasses.png"))
	plt.close()

	for label_idx, target_label in enumerate(labels):

		scores_to_plot = all_scores[:, label_idx]
		title = f"Distribution for '{target_label}'"
		plt.figure(figsize=(8, 4))
		sns.histplot(scores_to_plot, kde=True, bins=50)
		plt.title(title)
		plt.xlabel("Score")
		plt.ylabel("Frequency")
		plt.savefig(os.path.join(directory, f"{filename}_{target_label}_score.png"))
		plt.close()
	# Define bins starting at 0.05 up to 1.0 (step 0.05)
	bins = np.arange(0.05, 1.05, 0.05)
	bin_labels = [f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(len(bins) - 1)]

	# Collect binned counts for each target label
	stack_data = []

	for label_idx, target_label in enumerate(labels):
		scores_to_plot = all_scores[:, label_idx]
		counts, _ = np.histogram(scores_to_plot, bins=bins)
		stack_data.append(counts)

	stack_data = np.array(stack_data)  # shape: (num_labels, num_bins-1)

	# Plot stacked bar chart
	print(f"Plotting stacked bar chart for {model_name} - {filename}")
	fig, ax = plt.subplots(figsize=(10, 6))
	bottom = np.zeros(len(bin_labels))
	colors = sns.color_palette('tab20', len(labels))  # Optional: custom colors

	for i, (label, color) in enumerate(zip(labels, colors)):
		ax.bar(bin_labels, stack_data[i], bottom=bottom, label=label, color=color)
		bottom += stack_data[i]

	ax.set_xlabel("Score Range")
	ax.set_ylabel("Count")
	ax.set_title("Stacked Bar Chart of Sentiment Score Distributions")
	ax.legend(title="Sentiment")
	plt.xticks(rotation=45)
	plt.savefig(os.path.join(directory, f"{filename}_stacked.png"))
	plt.close()

	labels_list = list(distribution.keys())
	means = [np.mean(distribution[label]) for label in labels_list]
	stds = [np.std(distribution[label]) for label in labels_list]

	plt.figure(figsize=(8, 5))
	bars = plt.bar(labels_list, means, yerr=stds, capsize=8, color='skyblue')
	plt.ylabel("Mean Score")
	plt.title("Mean Sentiment Score per Label (with Std Dev)")
	plt.xticks(rotation=30)
	plt.savefig(os.path.join(directory, f"{filename}_mean-std-dev-per-label.png"))
	plt.close()

	# Define the mapping
	sentiment_weights = np.linspace(-1, 1, len(labels))

	# For each tweet's scores (softmax output), calculate the continuum score
	print(f"Plotting continuum scores for {model_name} - {filename}")
	continuum_scores = []
	for scores in all_scores:
		score = np.dot(sentiment_weights, scores)
		continuum_scores.append(score)

	plt.figure(figsize=(8, 4))
	plt.hist(continuum_scores, bins=100, color='skyblue', edgecolor='black')
	plt.title("Histogram of Sentiment Continuum Scores")
	plt.xlabel("Continuum Score (-1 = Strong Negative, 1 = Strong Positive)")
	plt.ylabel("Tweet Count")
	plt.tight_layout()
	plt.savefig(os.path.join(directory, f"{filename}_score-continuum.png"))
	plt.close()

	plt.figure(figsize=(8, 4))
	sns.kdeplot(continuum_scores, fill=True)
	plt.title("KDE of Sentiment Continuum Scores")
	plt.xlabel("Continuum Score (-1 = Strong Negative, 1 = Strong Positive)")
	plt.ylabel("Density")
	plt.tight_layout()
	plt.savefig(os.path.join(directory, f"{filename}_score-continuum-kde.png"))
	plt.close()

	plt.figure(figsize=(6, 4))
	plt.boxplot(continuum_scores, vert=False)
	plt.title("Boxplot of Sentiment Continuum Scores")
	plt.xlabel("Continuum Score (-1 = Strong Negative, 1 = Strong Positive)")
	plt.tight_layout()
	plt.savefig(os.path.join(directory, f"{filename}_score-continuum-boxplot.png"))
	plt.close()

	plt.figure(figsize=(6, 4))
	plt.boxplot(continuum_scores, vert=False)
	plt.title("Boxplot of Sentiment Continuum Scores")
	plt.xlabel("Continuum Score (-1 = Strong Negative, 1 = Strong Positive)")
	plt.tight_layout()
	plt.savefig(os.path.join(directory, f"{filename}_score-continuum-boxplot.png"))
	plt.close()

	plt.figure(figsize=(6, 4))
	sns.violinplot(x=continuum_scores)
	plt.title("Violin Plot of Sentiment Continuum Scores")
	plt.xlabel("Continuum Score (-1 = Strong Negative, 1 = Strong Positive)")
	plt.tight_layout()
	plt.xlim(-1, 1)
	plt.savefig(os.path.join(directory, f"{filename}_score-continuum-violin.png"))
	plt.close()

	print(f"Images saved to {directory}")


def dump_scores(tweet_scores, labels, model_name, out_dir, filename, verbose=False):
	directory = f"{out_dir}/scores/{model_name}/"
	if not os.path.exists(directory):
		os.makedirs(directory)

	print(f"Saving scores for {model_name} - {filename}")
	output_file = os.path.join(directory, f"{filename}.csv")
	with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['tweet'] + labels)  # Write header with labels
		iterator = tweet_scores.items()
		if verbose and tqdm:
			iterator = tqdm(iterator, total=len(tweet_scores), desc=f"Writing {os.path.basename(output_file)}", unit="rows")
		for tweet, scores in iterator:
			writer.writerow([tweet.strip()] + np.array2string(scores, separator=',', max_line_width=sys.maxsize)[1:-1].split(','))  # Convert numpy array to list for CSV
	print(f"Scores saved to {output_file}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="RoBERTa sentiment classification on tweet text files.")
	parser.add_argument("-i", "--input_file", required=True, help="Path to the input file or directory. If a directory is provided, all .txt files within will be processed.")
	parser.add_argument("-o", "--output_file", required=True, help="Path to the output directory. Output will be placed in this directory as /{model}/{csv|images}/{input_file}.{csv|png}.")
	parser.add_argument("--verbose", action="store_true", help="Show progress bars.")

	args = parser.parse_args()
	out_dir = args.output_file
	verbose = args.verbose
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	# is input a file or directory?
	files = []
	if os.path.isfile(args.input_file):
		files.append(args.input_file)
	else:
		for _root, _, _files in os.walk(args.input_file):
			for filename in _files:
				filepath = os.path.join(_root, filename)
				if filename.endswith('.txt'):
					files.append(filepath)

	for file in set(files):
		if verbose and tqdm:
			with open(file, 'r', encoding='utf-8') as f:
				lines = [line for line in tqdm(f, desc=f"Loading {os.path.basename(file)}", unit="lines")]
		else:
			with open(file, 'r', encoding='utf-8') as f:
				lines = f.readlines()

		for model_name, labels in models_labels.items():
			print(f"Running classification for model: {model_name} on file: {file}")
			tweet_scores, distribution, counts = run_classification(model_name, labels, lines, verbose=verbose)
			dump_scores(
				tweet_scores,
				list(labels.values()),
				model_name,
				out_dir,
				('_'.join(file.split('_')[:-1])).split('/')[-1],  # drop _tweets.txt and everything before /
				verbose=verbose
			)
			save_images(
				distribution,
				counts,
				np.array(list(tweet_scores.values())),
				list(labels.values()),
				model_name,
				out_dir,
				('_'.join(file.split('_')[:-1])).split('/')[-1]  # drop _tweets.txt and everything before /
			)
			print(f"Classification completed for model: {model_name} on file: {file}")
		print(f"All models processed for file: {file}")
	print("DONE!")
