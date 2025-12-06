import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np
from tqdm import tqdm
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance


def run_bertopic(
	documents_filtered, 
	min_cluster_size=0, 
	vectorized_documents=None,
	verbose=True,
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
		min_cluster_size=min_cluster_size,
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
		representation_model=representation_model
	)

	# Compute embeddings in batches
	print(f"Computing embeddings in batches of {batch_size}...")
	embeddings = []
	for i in tqdm(range(0, len(documents_filtered), batch_size), desc="Embedding batches"):
		batch = documents_filtered[i:i + batch_size]
		batch_emb = embedder.encode(batch, show_progress_bar=False, convert_to_numpy=True)
		embeddings.append(batch_emb)
	embeddings = np.vstack(embeddings)

	print("Fitting BERTopic with precomputed embeddings...")
	topics, _ = topic_model.fit_transform(
		documents_filtered, 
		embeddings, 
		vectorized_documents=vectorized_documents
	)
	initial_count = len(set([t for t in topics if t != -1]))
	print(f"Initial non-outlier topic count: {initial_count}")

	return topic_model, topics, embeddings


def cull_topics(topic_model, documents_filtered, TARGET_MAX=50):
	# ---- Reduce to at most 50 topics ----

	print(f"Reducing topics to {TARGET_MAX}...")
	topic_model.reduce_topics(documents_filtered, nr_topics=TARGET_MAX)

	final_count = len(set([t for t in topic_model.topics_ if t != -1]))
	print(f"Final non-outlier topic count: {final_count}")
	print(f"\nFound {len(set(topic_model.topics_))} topics!")


def save(documents, topic_model, topics, output_file_base="tweets_with_topics"):
	# Add topic assignments back to your dataframe
	print("Saving documents topic assignments...")
	with open(f"{output_file_base}_topic-mappings.csv", "w", encoding="utf-8") as f:
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
	output_file_base, 
	verbose = True,
	vectorised_documents=None,
	min_cluster_size=0,
	max_topics=50,
	):
	
	topic_model, topics, embeddings = run_bertopic(
		vectorised_documents=vectorised_documents,
		min_cluster_size=min_cluster_size,
		verbose=verbose,
	)
	cull_topics(
		topic_model, 
		vectorised_documents, 
		TARGET_MAX=max_topics
	)

	save(vectorised_documents, topic_model, topics, output_file_base)
	plot(topic_model, topics, embeddings, output_file_base)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="BERTopic Modeling on Tweets. Run from Topic Modelling multirunner.")
	# util.create_input_args(parser)
	# util.create_output_args(parser, suffix='{.csv|.png|.html}', help="Directory to save output plots and CSVs")
	# parser.add_argument("--document_column", default="tweet_text", help="Column name in the CSV that contains the tweet text")
	# parser.add_argument("--max_topics", type=int, default=50, help="Maximum number of topics to reduce to (default: 50)")
	# parser.add_argument("--min_cluster_size", type=int, default=150, help="Minimum cluster size for HDBSCAN (optional, helps control number of topics)")
	# parser.add_argument("--stop_words", required=False, help="Custom stopwords list to use")
	# parser.add_argument("--min-df", type=int, default=5, help="Minimum document frequency for CountVectorizer (default: 5)")
	# parser.add_argument("--max-df", type=float, default=0.7, help="Maximum document frequency ratio (default: 0.7)")

	# args = parser.parse_args()

	# if(args.stop_words):
	# 	print(f"Using custom stopwords from: {args.stop_words}")
	# 	custom_stopwords = load_stopwords(Path(args.stop_words))
	# 	print(f"Loaded {len(custom_stopwords)} custom stopwords")

	# input_files = util.parse_input_files_arg(args.input_file, ext="-denoised.txt")
	# output_files = util.parse_output_files_arg(args.output, input_files)

	# document_column = args.document_column

	# for input_file, output_file in zip(input_files, output_files):
	# 	process_file(
	# 		input_file,
	# 		output_file,
	# 		stopwords=custom_stopwords if args.stop_words else None,
	# 		verbose=True,
	# 		vectorised_documents=doc_term_matrix,
	# 	)
