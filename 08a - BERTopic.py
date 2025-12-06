import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import argparse
import re
import numpy as np
from tqdm import tqdm
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from pathlib import Path
import utilities as util

def preprocess_tweet(text):
	# Remove URLs
	text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

	# Remove user mentions
	text = re.sub(r'@\w+', '', text)

	# Remove hashtag symbol but keep the text
	text = re.sub(r'#(\w+)', r'\1', text)

	# Remove RT indicator
	text = re.sub(r'\bRT\b', '', text)

	# Remove extra whitespace
	text = ' '.join(text.split())

	return text.strip()


def load_csv(input_file, document_column='tweet_text'):
	# Load your CSV
	print("Loading CSV...")
	df = pd.read_csv(input_file, encoding='utf-8', on_bad_lines='skip', engine='python')
	print(f"Loaded {len(df)} tweets")

	# Extract the text column (replace 'text' with your column name)
	documents = df[document_column].tolist()

	return df, documents


def preprocess(documents):
	# Preprocess with progress bar
	print("Preprocessing tweets...")
	documents = [preprocess_tweet(doc) for doc in tqdm(documents, desc="Preprocessing")]

	# Optional: Apply preprocessing
	documents = [preprocess_tweet(doc) for doc in documents]

	# Remove empty/very short tweets
	documents_filtered = [doc for doc in documents if len(doc.split()) > 3]

	print(f"Filtered to {len(documents_filtered)} tweets with >3 words")
	return documents_filtered


def run_bertopic(documents_filtered, min_cluster_size=0, custom_stopwords=None):
	# ===== BERTopic Modeling =====
	print("Running BERTopic with batched embeddings...")

	# Parameters
	batch_size = 4096  # You can adjust this based on your GPU/CPU RAM
	embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

	# Getting rid of stopwords
	vectorizer_model = CountVectorizer(stop_words=custom_stopwords if custom_stopwords else 'english', lowercase=True, ngram_range=(1, 2))
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

	umap_model = UMAP(n_neighbors=15, n_components=5, metric='euclidean', low_memory=True)

	topic_model = BERTopic(
		embedding_model=None,  # We'll pass precomputed embeddings
		umap_model=umap_model,
		verbose=True,
		calculate_probabilities=False,
		hdbscan_model=hdbscan_model,
		vectorizer_model=vectorizer_model,
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
	topics, _ = topic_model.fit_transform(documents_filtered, embeddings)
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


def save(df, topic_model, topics, output_file_base="tweets_with_topics"):
	# Add topic assignments back to your dataframe
	print("Adding topics to dataframe...")
	df_filtered = df[df['tweet_text'].apply(lambda x: len(preprocess_tweet(x).split()) > 3)].copy()
	df_filtered['topic'] = topics

	# Save to CSV
	print(f"Saving results to {output_file_base}.csv...")
	df_filtered.to_csv(f"{output_file_base}.csv", index=False)
	print("Done.")
	print("Saving topic info and model...")
	topic_info = topic_model.get_topic_info()
	topic_info.to_csv(f"{output_file_base}-topic_info.csv", index=False)
	# topic_model.save(f"{output_file_base}-bertopic_model_50max")
	print("Done.")


def plot(topic_model, topics, embeddings, output_file_base="tweets_with_topics"):
	final_count = len(set([t for t in topics if t != -1]))
	# ===== PLOTTING =====
	print("Generating visualizations (50 max)...")
	# Barchart (show all final topics if <= 50)
	top_n = min(50, final_count)
	fig_bar = topic_model.visualize_barchart(top_n_topics=top_n)
	fig_bar.write_html(f"{output_file_base}-topic_barchart.html")

	fig_topics = topic_model.visualize_topics()
	fig_topics.write_html(f"{output_file_base}-topic_visualization.html")

	fig_hier = topic_model.visualize_hierarchy()
	fig_hier.write_html(f"{output_file_base}-topic_hierarchy.html")

	topic_info = topic_model.get_topic_info()
	print(topic_info.head(15))


def load_stopwords(custom_path: Path | None):
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

def process_file(input_file, output_dir, stopwords, verbose = True):
	if input_file.endswith('.csv'):
		df, documents = load_csv(input_file, document_column=document_column)
	else:
		with open(input_file, 'r', encoding='utf-8') as f:
			documents = f.readlines()
		# create dataframe
		df = pd.DataFrame({document_column: documents})
	documents_filtered = preprocess(documents)
	topic_model, topics, embeddings = run_bertopic(documents_filtered, min_cluster_size=args.min_cluster_size, custom_stopwords=custom_stopwords if args.stop_words else None)
	cull_topics(topic_model, documents_filtered, TARGET_MAX=args.max_topics)

	save(df, topic_model, topics, output_file_base=output_dir)
	plot(topic_model, topics, embeddings, output_file_base=output_dir)


if __name__ == "__main__":
	# TODO: This should take a csv apparently
	# TODOTODO: MAke it take either a csv or text files
	parser = argparse.ArgumentParser(description="BERTopic Modeling on Tweets")
	util.create_input_args(parser)
	util.create_output_args(parser, suffix='{.csv|.png|.html}', help="Directory to save output plots and CSVs")
	parser.add_argument("--document_column", default="tweet_text", help="Column name in the CSV that contains the tweet text")
	parser.add_argument("--max_topics", type=int, default=50, help="Maximum number of topics to reduce to (default: 50)")
	parser.add_argument("--min_cluster_size", type=int, default=150, help="Minimum cluster size for HDBSCAN (optional, helps control number of topics)")
	parser.add_argument("--stop_words", required=False, help="Custom stopwords list to use")

	args = parser.parse_args()

	if(args.stop_words):
		print(f"Using custom stopwords from: {args.stop_words}")
		custom_stopwords = load_stopwords(Path(args.stop_words))
		print(f"Loaded {len(custom_stopwords)} custom stopwords")

	input_files = util.parse_input_files_arg(args.input_file, ext="-denoised.txt")
	output_files = util.parse_output_files_arg(args.output, input_files)

	document_column = args.document_column

	for input_file in args.input_file:
		process_file(
			input_file,
			output_dir=args.output,
			stopwords=custom_stopwords if args.stop_words else None,
			verbose=True
		)
