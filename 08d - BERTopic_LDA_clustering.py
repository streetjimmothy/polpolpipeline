import argparse
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation
import keras
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

class Autoencoder:
	"""
	Autoencoder for learning latent space representation
	architecture simplified for only one hidden layer
	"""

	def __init__(self, latent_dim=32, activation='relu', epochs=200, batch_size=128):
		self.latent_dim = latent_dim
		self.activation = activation
		self.epochs = epochs
		self.batch_size = batch_size
		self.autoencoder = None
		self.encoder = None
		self.decoder = None
		self.his = None

	def _compile(self, input_dim):
		'''
		compile the computational graph
		'''
		input_vec = keras.layers.Input(shape=(input_dim,))
		encoded = keras.layers.Dense(self.latent_dim, activation=self.activation)(input_vec)
		decoded = keras.layers.Dense(input_dim, activation=self.activation)(encoded)
		self.autoencoder = keras.models.Model(input_vec, decoded)
		self.encoder = keras.models.Model(input_vec, encoded)
		encoded_input = keras.layers.Input(shape=(self.latent_dim,))
		decoded_layer = self.autoencoder.layers[-1]
		self.decoder = keras.models.Model(encoded_input, self.autoencoder.layers[-1](encoded_input))
		self.autoencoder.compile(optimizer='adam', loss=keras.losses.mean_absolute_error)

	def fit(self, X):
		if not self.autoencoder:
			self._compile(X.shape[1])
		X_train, X_test = train_test_split(X)
		self.his = self.autoencoder.fit(X_train, X_train, epochs=200, batch_size=92, shuffle=True, validation_data=(X_test, X_test), verbose=0)

def lda_topic_assignments(matrix, n_topics, max_iter=100):
	print("Fitting LDA model...")
	lda = LatentDirichletAllocation(
		n_components=n_topics,
		max_iter=max_iter,
		learning_method="batch",  # TODO: Consider online? might be faster for large corpora
		evaluate_every=0,
		verbose=1,
		n_jobs=-1,
	)
	lda_topics = lda.fit_transform(matrix, )
	topic_assignments = np.argmax(lda_topics, axis=1)
	return topic_assignments, lda


def get_feature_names(vectorizer):
	if hasattr(vectorizer, "get_feature_names_out"):
		return vectorizer.get_feature_names_out().tolist()
	return vectorizer.get_feature_names()

def run_bertopic(
	documents,
	term_freq_matrix=None,
	vectoriser=None,
	min_cluster_size=0,
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
		embeddings
	)

	initial_count = len(set([t for t in topics if t != -1]))
	print(f"Initial non-outlier topic count: {initial_count}")

	return topic_model, topics, embeddings


def cull_topics(topic_model, documents, TARGET_MAX=50):
	# ---- Reduce to at most 50 topics ----

	print(f"Reducing topics to {TARGET_MAX}...")
	topic_model.reduce_topics(documents, nr_topics=TARGET_MAX)  # TODO: Try use_ctfidf=True

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


def plot(
	documents,
	BERT_model, 
	BERT_topics,
	LDA_topic_assignments,
	combined_representation,
	output_file_base="tweets_with_topics"
):
	# ===== PLOTTING =====
	pca = PCA(n_components=2)
	vis_2d = pca.fit_transform(combined_representation)
	print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

	# Create dataframe for easier plotting
	df = pd.DataFrame({
		'x': vis_2d[:, 0],
		'y': vis_2d[:, 1],
		'topic': BERT_topics,
		'document': documents[:len(BERT_topics)]
	})

	# Plot: each topic gets a different color
	fig = px.scatter(
		df,
		x='x',
		y='y',
		color='topic',
		hover_data=['document'],
		title='Topic Structure (BERTopic colors on fused latent space)',
		labels={'x': 'PCA/t-SNE Dim 1', 'y': 'PCA/t-SNE Dim 2', 'topic': 'BERTopic Topic'},
		height=700
	)

	output_path = Path(output_file_base + "BERT_cluster_plot.png")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.update_traces(marker=dict(size=5, opacity=0.7))
	fig.write_html("topic_structure_viz.html")

	# Get top words for each topic from BERTopic
	topic_info = BERT_model.get_topic_info()

	# Create a label dict: topic_id -> top 3 words
	labels = {}
	for _, row in topic_info.iterrows():
		topic_id = row['Topic']
		words = row['Name']  # BERTopic already formats this as "word_word_word"
		labels[topic_id] = words

	df['topic_label'] = df['topic'].map(labels).fillna('Outlier')

	# Re-plot with labels
	fig = px.scatter(
		df,
		x='x',
		y='y',
		color='topic_label',
		hover_data=['document', 'topic'],
		title='Topic Structure with Labels',
		height=700
	)
	fig.write_html("topic_structure_labeled.html")

	df['lda_topic'] = LDA_topic_assignments[:len(BERT_topics)]	#TODO: should this be len of LDA topics?


	fig_lda = px.scatter(
		df,
		x='x',
		y='y',
		color='lda_topic',
		title='Same latent space, colored by LDA topics',
		height=700
	)
	fig_lda.write_html("topic_structure_lda_colored.html")

	# print("Generating visualizations (50 max)...")
	# # Barchart (show all final topics if <= 50)
	# top_n = min(50, final_count)
	# fig_bar = topic_model.visualize_barchart(top_n_topics=top_n)
	# fig_bar.write_html(f"{output_file_base}_topic-barchart.html")

	# fig_topics = topic_model.visualize_topics()
	# fig_topics.write_html(f"{output_file_base}_topic-visualization.html")

	# fig_hier = topic_model.visualize_hierarchy()
	# fig_hier.write_html(f"{output_file_base}_topic-hierarchy.html")

	# topic_info = topic_model.get_topic_info()
	# print(topic_info.head(15))


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


	LDA_topics, LDA_model = lda_topic_assignments(
		term_freq_matrix,
		num_topics,
		max_iter=max_iterations
	)

	# Get LDA soft probabilities (not just hard assignments)
	lda_probs = LDA_model.transform(term_freq_matrix)  # (n_docs, n_topics)

	BERT_model, BERT_topics, embeddings = run_bertopic(
		documents,
		term_freq_matrix=term_freq_matrix,
		vectoriser=vectoriser,
		min_cluster_size=min_cluster_size,
		verbose=verbose,
	)
	cull_topics(
		BERT_model,
		documents,
		TARGET_MAX=num_topics
	)

	# Combine LDA probabilities and BERT embeddings
	gamma = 0.5  # Weight to balance LDA vs BERT influence
	combined = np.c_[lda_probs * gamma, embeddings]  # (n_docs, n_topics + embedding_dim)
	
	# Pass through autoencoder for dimensionality reduction
	print("Fitting autoencoder on combined representation...")
	ae = Autoencoder(latent_dim=64, epochs=200)
	ae._compile(combined.shape[1])
	ae.fit(combined)
	
	# Get final fused representation
	combined_representation = ae.encoder.predict(combined)  # (n_docs, latent_dim)

	save(documents, BERT_model, BERT_topics, output_file_base)
	plot(
		documents,
		BERT_model, 
		BERT_topics,
		LDA_topics,
		combined_representation, 
		output_file_base
	)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Process tweet text files combining BERT and LDA using Kmeans clustering to merge results.Run as part of a multirunner pipeline.")
# 	parser.add_argument("-i", "--input_file", "--input_files", "--input_dir", nargs='+', required=True, help="Path to the input text file containing tweets (multiple files can be specified, they will be process in turn. A directory can also be specified, in which case all txt files in that directory will be used)")
# 	parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
# 	args = parser.parse_args()

# 	process_files(Path(args.input_dir), Path(args.domain_csv), args.unicode_mode, args.verbose)
# #parser = argparse.ArgumentParser(description='contextual_topic_identification tm_test:1.0')

# #parser.add_argument('--fpath', default='/kaggle/working/train.csv')
# #parser.add_argument('--ntopic', default=10,)
# #parser.add_argument('--method', default='TFIDF')
# #parser.add_argument('--samp_size', default=20500)

# #args = parser.parse_args()

