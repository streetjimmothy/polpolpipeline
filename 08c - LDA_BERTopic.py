import argparse
import os
from bertopic import BERTopic
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from pathlib import Path
import matplotlib.pyplot as plt

def read_documents(input_file):
	with open(input_file, "r", encoding="utf-8") as f:
		docs = [line.strip() for line in f if line.strip()]
	return docs


def load_stopwords(custom_path):
	stopwords = set(ENGLISH_STOP_WORDS)
	if custom_path:
		if not custom_path.is_file():
			raise FileNotFoundError(f"Custom stopword file not found: {custom_path}")
		with custom_path.open("r", encoding="utf-8", errors="ignore") as f:
			for line in f:
				token = line.strip().lower()
				if token:
					stopwords.add(token)
	return sorted(stopwords)


def lda_topic_assignments(matrix, n_topics, max_iter=100):
	lda = LatentDirichletAllocation(
		n_components=n_topics,
		max_iter=max_iter,
		learning_method="batch",  # TODO: Consider online? might be faster for large corpora
		evaluate_every=0,
		verbose=1,
		n_jobs=-1,
	)
	lda_topics = lda.fit_transform(matrix)
	topic_assignments = np.argmax(lda_topics, axis=1)
	return topic_assignments, lda

def bertopic_with_custom_topics(docs, topic_assignments):
	topic_model = BERTopic()
	topics, probs = topic_model.fit_transform(docs, y=topic_assignments)
	return topic_model, topics

def plot_dendrogram(topic_model, output_dir):
	embeddings = topic_model.c_tf_idf_.toarray()
	linked = linkage(embeddings, 'ward')
	plt.figure(figsize=(10, 7))
	dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
	plt.title("Topic Dendrogram")
	output_path = Path(output_dir + "lda_bert_dendrogram.png")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path)
	plt.close()

def plot_clusters(topic_model, output_dir):
	fig = topic_model.visualize_topics()
	output_path = Path(output_dir + f"clusters.png"))
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.write_image(output_path)

def plot_top_words(topic_model, output_dir, topic_id=0, n_words=10):
	words, scores = zip(*topic_model.get_topic(topic_id)[:n_words])
	plt.figure(figsize=(8, 5))
	plt.bar(words, scores)
	plt.title(f"Top {n_words} Words in Topic {topic_id}")
	plt.xticks(rotation=45)
	plt.tight_layout()
	output_path = Path(output_dir + f"topic_{topic_id}_words.png"))
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path)
	plt.close()


def run(
	documents,
	output_file_base,
	term_freq_matrix=None,
	vectoriser=None,
	min_cluster_size=0,
	num_topics=50,
	max_iterations=100,
	verbose=True,
):
	topic_assignments, lda = lda_topic_assignments(
		term_freq_matrix, 
		num_topics,
		max_iter=max_iterations
	)
	topic_model, topics = bertopic_with_custom_topics(documents, topic_assignments)
	plot_dendrogram(topic_model, output_file_base)
	plot_clusters(topic_model, output_file_base)
	plot_top_words(topic_model, output_file_base, topic_id=0, n_words=10)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="LDA-powered BERTopic pipeline. Run from Topic Modelling multirunner.")