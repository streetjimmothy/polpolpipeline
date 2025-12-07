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
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
from tqdm import tqdm

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

def cull_topics(topic_model, documents, TARGET_MAX=50):
	# ---- Reduce to at most 50 topics ----

	print(f"Reducing topics to {TARGET_MAX}...")
	topic_model.reduce_topics(documents, nr_topics=TARGET_MAX) #TODO: Try use_ctfidf=True

	final_count = len(set([t for t in topic_model.topics_ if t != -1]))
	print(f"Final non-outlier topic count: {final_count}")
	print(f"\nFound {len(set(topic_model.topics_))} topics!")

def bertopic_with_custom_topics(
	documents, 
	topic_assignments,
	min_cluster_size=0, 
	verbose=True
):
	# ===== BERTopic Modeling =====
	print("Running BERTopic with batched embeddings...")

	# Parameters
	batch_size = 256  # You can adjust this based on your GPU/CPU RAM
	embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

	ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

	# diversify words in each topic such that we limit the number of duplicate words we find in each topic.
	representation_model = MaximalMarginalRelevance(diversity=0.2)

	# Use HDBSCAN with a minimum cluster size to control number of topics
	hdbscan_model = HDBSCAN(
		min_cluster_size=int(min_cluster_size),
		metric='euclidean',
		cluster_selection_method='eom',
		prediction_data=False
	)

	umap_model = UMAP(n_neighbors=15, n_components=5, metric='euclidean')

	topic_model = BERTopic(
		embedding_model=None,  # We'll pass precomputed embeddings
		umap_model=umap_model,
		verbose=verbose,
		calculate_probabilities=False,
		hdbscan_model=hdbscan_model,
		ctfidf_model=ctfidf_model,
		representation_model=representation_model,
	)

	# Compute embeddings in batches
	print(f"Computing embeddings in batches of {batch_size}...")
	embeddings = []
	for i in tqdm(range(0, len(documents), batch_size), desc="Embedding batches"):
		batch = documents[i:i + batch_size]
		batch_emb = embedder.encode(batch, show_progress_bar=False, convert_to_numpy=True)
		embeddings.append(batch_emb)
	embeddings = np.vstack(embeddings)

	print("Fitting BERTopic with precomputed embeddings...")
	topics, _ = topic_model.fit_transform(
		documents, 
		embeddings,
		y=topic_assignments
	)
	
	initial_count = len(set([t for t in topics if t != -1]))
	print(f"Initial non-outlier topic count: {initial_count}")

	return topic_model, topics, embeddings

def cull_topics(topic_model, documents, TARGET_MAX=50):
	# ---- Reduce to at most 50 topics ----

	print(f"Reducing topics to {TARGET_MAX}...")
	topic_model.reduce_topics(documents, nr_topics=TARGET_MAX) #TODO: Try use_ctfidf=True

	final_count = len(set([t for t in topic_model.topics_ if t != -1]))
	print(f"Final non-outlier topic count: {final_count}")
	print(f"\nFound {len(set(topic_model.topics_))} topics!")


def save(documents, topic_model, topics, output_file_base="tweets_with_topics"):
	# Add topic assignments back to your dataframe
	print("Saving documents topic assignments...")
	output_path = Path(f"{output_file_base}_with_topics.csv")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as f:
		f.write("document_index,document, topic\n")
		for i, topic in tqdm(enumerate(topics), total=len(topics), desc="Saving topic assignments"):
			f.write(f"{i},{documents[i]},{topic}\n")

	# Save to CSV
	print("Saving topic info and model...")
	topic_info = topic_model.get_topic_info()
	topic_info.to_csv(f"{output_file_base}-topic_info.csv", index=False)
	# topic_model.save(f"{output_file_base}-bertopic_model_50max")
	print("Done.")


def plot(topic_model, topics, output_file_base="tweets_with_topics"):
	final_count = len(set([t for t in topics if t != -1]))
	# ===== PLOTTING =====
	print("Generating visualizations (50 max)...")
	# Barchart (show all final topics if <= 50)
	top_n = min(50, final_count)
	fig_bar = topic_model.visualize_barchart(top_n_topics=top_n)
	fig_bar.write_html(f"{output_file_base}_topic-barchart.html")

	fig_topics = topic_model.visualize_topics()
	fig_topics.write_html(f"{output_file_base}_topic-visualization.html")

	fig_hier = topic_model.visualize_hierarchy()
	fig_hier.write_html(f"{output_file_base}_topic-hierarchy.html")

	topic_info = topic_model.get_topic_info()
	print(topic_info.head(15))

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
	topic_model, topics, embeddings = bertopic_with_custom_topics(
		documents, 
		topic_assignments,
		min_cluster_size=min_cluster_size,
		verbose=verbose
	)
	cull_topics(
		topic_model, 
		documents, 
		TARGET_MAX=num_topics
	)

	save(documents, topic_model, topics, output_file_base)
	plot(topic_model, topics, output_file_base)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="LDA-powered BERTopic pipeline. Run from Topic Modelling multirunner.")