import argparse
import utilities as utils
from collections import defaultdict

def plot_with_BERTopic(topic_model, output_file_base="tweets_with_topics"):
	#cap at 10 topics for visualization
	topic_model

	print("Generating BERTopic visualizations...")
	fig_bar = topic_model.visualize_barchart()
	fig_bar.write_html(f"{output_file_base}_topic-barchart.html")

	fig_topics = topic_model.visualize_topics(n_words=10)
	fig_topics.write_html(f"{output_file_base}_topic-visualization.html")

	fig_hier = topic_model.visualize_hierarchy()
	fig_hier.write_html(f"{output_file_base}_topic-hierarchy.html")

	print("Visualizations saved.")

def plot_from_csv(topic_info_csv: str, document_topic_csv: str, output_file_base="tweets_with_topics"):
	import pandas as pd
	import numpy as np
	import ast
	from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
	from tqdm import tqdm
	import matplotlib.pyplot as plt

	stopwords = list(ENGLISH_STOP_WORDS) + ["rt", "https", "co", "amp", "com", "au"]

	print(f"Loading BERTopic topic info from CSV: {topic_info_csv}")
	info_df = pd.read_csv(topic_info_csv)
	topic_ids = info_df['Topic'].astype(int).tolist()
	counts = info_df['Count'].astype(int).tolist()
	representations = info_df['Representation'].apply(ast.literal_eval).tolist()
	names = []
	for idx, topic_id in enumerate(topic_ids):
		if topic_id == -1:
			names.append("Outlier")
		else:
			name = f"Topic {topic_id:02d} ({counts[idx]} docs): "
			for term in representations[idx]:
				if term not in stopwords and term.isalpha():
					if len(term) > 20:
						term = term[:17] + "..."  # truncate long terms
					name += term	
			names.append(name)
		
	print(f"Loading document-topic assignments from CSV: {document_topic_csv}")
	
	doc_topic_df = pd.read_csv(document_topic_csv)
	docs_by_topic = defaultdict(list)
	for index, row in tqdm(doc_topic_df.iterrows(), total=doc_topic_df.shape[0], desc="Processing documents"):
		topic = row['topic']
		if topic in topic_ids:
			topic_name = names[topic_ids.index(topic)]
		else:
			topic_name = "Outlier"
		docs_by_topic[topic_name].append(row['document'])
	topic_names = list(docs_by_topic.keys())

	#we do this because Python string handling is slooooow
	for topic_name in docs_by_topic:
		docs_by_topic[topic_name] = "\n".join(docs_by_topic[topic_name])

	print("Generating TF-IDF matrix for topics...")
	vectorizer = TfidfVectorizer(
		stop_words='english', 
		max_df=0.75,
		token_pattern=r'(?u)\b[^\d\W]+\b'  # no digits
		#sublinear_tf=True)  # sublinear_tf=True downweights high raw term frequencies
	)
	tfidf_matrix = vectorizer.fit_transform(list(docs_by_topic.values()))

	#get the top terms per topic
	terms = vectorizer.get_feature_names_out()
	topic_terms_scores = {}
	for idx, row in enumerate(tfidf_matrix):
		row = row.toarray().flatten()
		top_indices = row.argsort()[-10:][::-1]
		top_terms = [terms[i] for i in top_indices]
		topic_terms_scores[topic_names[idx]] = {}
		for i in top_indices:
			topic_terms_scores[topic_names[idx]][terms[i]] = row[i]
			

	#plot
	print("Generating topic-term matrix visualization...")
	fig, axes = plt.subplots(3, 4, figsize=(8 * 3, 4 * 4), facecolor='w')
	axes = axes.flatten()

	topic_names_sorted = sorted(topic_terms_scores.keys())
	colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(topic_terms_scores.keys())))
	for idx, topic_name in enumerate(topic_names_sorted):
		if idx >= len(axes):
			break  # only plot up to available subplots
		print(f"Plotting topic: {topic_name}")
		terms = list(topic_terms_scores[topic_name].keys())
		scores = list(topic_terms_scores[topic_name].values())
		#nmormalise the scores to 0-1
		max_score = max(scores)
		scores = [s / max_score for s in scores]

		axes[idx].barh(terms, scores, color=colors[idx])
		axes[idx].set_xlabel('TF-IDF Score')
		if ' (' in topic_name:
			axes[idx].set_title(f"Top Terms for {topic_name.split(')')[0]})")
		else:
			axes[idx].set_title(f'Top Terms for {topic_name}')
		axes[idx].invert_yaxis()  # highest scores on top
		
		topic_filename = topic_name[:8]
	path = "\\".join(topic_info_csv.split('\\')[:-1])
	path += f'\\{output_file_base}.png'
	print(f"Saving topic-term visualization to: {path}")
	plt.tight_layout()
	plt.savefig(path)
	plt.close()
	
	
	
def main():
	parser = argparse.ArgumentParser(description="BERTopic Visualization Runner")
	utils.create_input_args(parser)

	args = parser.parse_args()


	input_files = utils.parse_input_files_arg(args.input_file, ext=".csv")
	if len(input_files) == 1:
		from bertopic import BERTopic
		print(f"Loading BERTopic model from: {input_files[0]}")
		topic_model = BERTopic.load(input_files[0])
		plot_with_BERTopic(topic_model, output_file_base="bertopic_model_viz")
	else:
		
		if len(input_files) != 2:
			raise SystemExit("Please provide exactly two input CSV files: one for topic info and one for document-topic assignments.")
		topic_info_csv = input_files[0]
		docment_topic_csv = input_files[1]
		print(f"Loading topic info from: {topic_info_csv}")
		print(f"Loading document-topic assignments from: {docment_topic_csv}")
		plot_from_csv(topic_info_csv, docment_topic_csv, output_file_base="bertopic_csv_viz")

if __name__ == "__main__":
	main()