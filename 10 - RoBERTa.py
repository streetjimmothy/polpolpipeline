from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import os
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

	batch_size = 512 

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
	"./models/cardiffnlp/twitter-roberta-large-emotion-latest":
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
	"./models/cardiffnlp/twitter-roberta-base-sentiment-latest":
		{
			0: "negative",
			1: "neutral",
			2: "positive"
		},
	"./models/cardiffnlp/twitter-roberta-large-topic-sentiment-latest":
		{
			0: "strongly negative",
			1: "negative",
			2: "negative or neutral",
			3: "positive",
			4: "strongly positive"
		},
	"./models/cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual":  # primarily for ARG, but applied uniformly
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
	"./models/dccuchile/bert-base-spanish-wwm-cased":
		{
			0: "negative",
			1: "positive"
		},
	"./models/VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis":
		{
			0: "negative",
			1: "positive"
		}
}


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
				if filename.endswith('-denoised.txt'):
					files.append(filepath)

	print(f"Found {len(files)} files to process.")
	for file in set(files):
		print(f"Processing file: {file}")
		if verbose and tqdm:
			with open(file, 'r', encoding='utf-8') as f:
				lines = [line for line in tqdm(f, desc=f"Loading {os.path.basename(file)}", unit="lines")]
		else:
			with open(file, 'r', encoding='utf-8') as f:
				lines = f.readlines()
		
		# Deduplicate lines
		lines = list(dict.fromkeys(lines))
		
		print(f"Loaded {len(lines)} unique lines from {file}.")

		for model_name, labels in models_labels.items():
			print(f"Running classification for model: {model_name} on file: {file}")
			tweet_scores, distribution, counts = run_classification(model_name, labels, lines, verbose=verbose)
			dump_scores(
				tweet_scores,
				list(labels.values()),
				model_name,
				out_dir,
				('-'.join(file.split('-')[:2])).split('/')[-1],  # keep everything before the second '-' and everything after the last '/'
				verbose=verbose
			)
			print(f"Classification completed for model: {model_name} on file: {file}")
		print(f"All models processed for file: {file}")
	print("DONE!")
