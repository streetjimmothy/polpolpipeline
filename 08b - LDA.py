#!/usr/bin/env python3
"""GPU-accelerated LDA topic modeling pipeline with visualization outputs."""
import argparse
import os
from pathlib import Path
from typing import Iterable, List, Sequence

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from cuml.decomposition import LatentDirichletAllocation

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tqdm.auto import tqdm

import utilities as util


def fit_lda(
	matrix: cp.ndarray, 
	n_topics: int, 
	max_iter: int
) -> LatentDirichletAllocation:
	print(f"Fitting LDA with {n_topics} topics, max {max_iter} iterations...")
	with tqdm(total=max_iter, desc="LDA training", unit="iteration") as pbar:
		class ProgressCallback:
			def __init__(self, pbar):
				self.pbar = pbar
				self.prev_iter = 0
			
			def __call__(self, model):
				# Update progress based on current iteration
				current_iter = model.n_iter_
				if current_iter > self.prev_iter:
					self.pbar.update(current_iter - self.prev_iter)
					self.prev_iter = current_iter
		
		callback = ProgressCallback(pbar)
		lda = LatentDirichletAllocation(
			n_components=n_topics,
			max_iter=max_iter,
			learning_method="batch",	#TODO: Consider online? might be faster for large corpora
			evaluate_every=1,
			verbose=0,
		)
		lda.fit(matrix)
		pbar.update(max_iter - pbar.n)  # Ensure full progress bar
	return lda


def get_feature_names(vectorizer) -> List[str]:
	if hasattr(vectorizer, "get_feature_names_out"):
		return vectorizer.get_feature_names_out().tolist()
	return vectorizer.get_feature_names()


def plot_dendrogram(topic_word_matrix: np.ndarray, output_dir: Path) -> Path:
	labels = [f"Topic {i}" for i in range(topic_word_matrix.shape[0])]
	distance_matrix = pdist(topic_word_matrix, metric="cosine")
	linkage_matrix = linkage(distance_matrix, method="ward")
	fig, ax = plt.subplots(figsize=(10, 6))
	dendrogram(linkage_matrix, labels=labels, ax=ax)
	ax.set_title("Topic Similarity Dendrogram")
	ax.set_ylabel("Cosine distance")
	output_path = output_dir / "lda_dendrogram.png"
	fig.tight_layout()
	fig.savefig(output_path, dpi=200)
	plt.close(fig)
	return output_path


def plot_cluster(topic_word_matrix: np.ndarray, output_dir: Path) -> Path:
	pca = PCA(n_components=2, random_state=0)
	coords = pca.fit_transform(topic_word_matrix)
	fig, ax = plt.subplots(figsize=(8, 6))
	ax.scatter(coords[:, 0], coords[:, 1], c=np.arange(coords.shape[0]), cmap="tab10")
	for idx, (x, y) in enumerate(coords):
		ax.text(x + 0.01, y + 0.01, f"T{idx}", fontsize=9)
	ax.set_title("Topic Clusters (PCA)")
	ax.set_xlabel("PC 1")
	ax.set_ylabel("PC 2")
	output_path = output_dir / "lda_cluster_plot.png"
	fig.tight_layout()
	fig.savefig(output_path, dpi=200)
	plt.close(fig)
	return output_path


def plot_top_words(
	topic_word_matrix: np.ndarray,
	feature_names: Sequence[str],
	top_n: int,
	output_dir: Path,
) -> Path:
	topics = topic_word_matrix.shape[0]
	cols = min(3, topics)
	rows = int(np.ceil(topics / cols))
	fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

	for topic_id in range(topics):
		row, col = divmod(topic_id, cols)
		ax = axes[row][col]
		top_indices = np.argsort(topic_word_matrix[topic_id])[::-1][:top_n]
		words = [feature_names[i] for i in top_indices]
		scores = topic_word_matrix[topic_id][top_indices]
		ax.barh(words[::-1], scores[::-1], color="steelblue")
		ax.set_title(f"Topic {topic_id}")
		ax.set_xlabel("Frequency")

	# Hide unused subplots
	for idx in range(topics, rows * cols):
		row, col = divmod(idx, cols)
		axes[row][col].set_visible(False)

	fig.tight_layout()
	output_path = output_dir / "lda_top_words.png"
	fig.savefig(output_path, dpi=200)
	plt.close(fig)
	return output_path


def run(
	documents,
	output_file_base,
	term_freq_matrix=None,
	vectoriser=None,
	num_topics=50,
	max_iterations=1000,
	verbose=True,
):
	if len(documents) < num_topics:
		raise ValueError(f"Number of documents ({len(documents)}) must be >= number of topics ({num_topics}).")

	lda_model = fit_lda(
		term_freq_matrix, 
		n_topics=num_topics, 
		max_iter=max_iterations
	)

	# Convert GPU arrays to CPU for plotting
	topic_word = cp.asnumpy(lda_model.components_)
	feature_names = get_feature_names(vectoriser)

	dendro_path = plot_dendrogram(topic_word, output_file_base)
	cluster_path = plot_cluster(topic_word, output_file_base)
	bars_path = plot_top_words(topic_word, feature_names, 5, output_file_base)

	print("Generated plots:")
	print(f"  Dendrogram:	{dendro_path}")
	print(f"  Cluster plot:  {cluster_path}")
	print(f"  Top words:	 {bars_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run GPU-accelerated LDA on a line-delimited text corpus and generate a dendrogram, cluster plot, and per-topic top-word bar charts. Run from Topic Modelling multirunner.")
	# util.create_input_args(parser)
	# util.create_output_args(parser, suffix='{.csv|.png|.html}', help="Directory to save output plots and CSVs")
	# parser.add_argument("--num_topics", "-t",type=int,required=True,help="Number of LDA topics to learn")
	# parser.add_argument("--stop_words", required=False, help="Custom stopwords list to use")
	# parser.add_argument("--max-iter", type=int, default=1000, help="Maximum number of LDA iterations (default: 1000)")
	# parser.add_argument("--min-df", type=int, default=5, help="Minimum document frequency for CountVectorizer (default: 5)")
	# parser.add_argument("--max-df", type=float, default=0.7, help="Maximum document frequency ratio (default: 0.7)")

	# args = parser.parse_args()

	# if (args.stop_words):
	# 	print(f"Using custom stopwords from: {args.stop_words}")
	# 	custom_stopwords = load_stopwords(Path(args.stop_words))
	# 	print(f"Loaded {len(custom_stopwords)} custom stopwords")

	# input_files = util.parse_input_files_arg(args.input_file, ext="-denoised.txt")
	# output_files = util.parse_output_files_arg(args.output, input_files)

	# document_column = args.document_column

	# for input_file, output_file in zip(input_files, output_files):
	# 	docs = load_documents(input_file)

		

		# vectorizer, doc_term_matrix = train_vectorizer(
		# 	docs,
		# 	stopwords=custom_stopwords if args.stop_words else None,
		# 	min_df=args.min_df,
		# 	max_df=args.max_df,
		# )

		# process_file(
		# 	input_file,
		# 	output_file,
		# 	stopwords=custom_stopwords if args.stop_words else None,
		# 	verbose=True,
		# 	args
		# )
