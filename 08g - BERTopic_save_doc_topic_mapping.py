import sys, os, csv
from bertopic import BERTopic
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer


def load_stopwords(custom_path: Path | None):
	stopwords = list(ENGLISH_STOP_WORDS)
	if custom_path:
		if not custom_path.is_file():
			raise FileNotFoundError(f"Custom stopword file not found: {custom_path}")
		with custom_path.open("r", encoding="utf-8", errors="ignore") as f:
			for line in f:
				token = line.strip().lower()
				if token:
					stopwords.append(token)
	return sorted(stopwords)


def load_documents(path: str, document_column: str = 'tweet_text'):
	if not os.path.isfile(path) and not os.path.isdir(path):
		raise FileNotFoundError(f"Input file not found: {path}")

	documents = []
	if str(path).endswith('.csv'):
		print("Loading CSV ...")
		with open(path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
			reader = csv.DictReader(f)
			for row in tqdm(reader, desc=f"Extracting {document_column}", unit="row"):
				text = row.get(document_column, None)
				if text is not None:
					documents.append(text)
	else:
		with open(path, 'r', encoding='utf-8') as f:
			for line in tqdm(f, desc=f"Loading {path}", unit="line"):
				doc = line.strip()
				if doc:
					documents.append(doc)
	print(f"Loaded {len(documents)} rows from {path}")
	return documents


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
		f.write("document_index,document,topic\n")
		for i, topic in tqdm(enumerate(topics), total=len(topics), desc="Saving topic assignments"):
			f.write(f"{i},{documents[i]},{topic}\n")

	# Save to CSV
	print("Saving topic info and model...")
	topic_info = topic_model.get_topic_info()
	topic_info.to_csv(f"{output_file_base}-topic_info.csv", index=False)
	# topic_model.save(f"{output_file_base}-bertopic_model_50max")
	print("Done.")

print("Starting BERTopic document-topic mapping save...")
print("Loading BERTopic model...")
print(f"Model file: {sys.argv[1]}")
topic_model = BERTopic.load(sys.argv[1])


print(f"Using custom stopwords from: stopwords.txt")
stopwords = load_stopwords(Path("stopwords.txt"))

vectoriser = CountVectorizer(
    stop_words=stopwords,
    lowercase=True,
    min_df=5,
    max_df=0.7,
    binary=False,
    ngram_range=(1, 2),
)

documents = load_documents(sys.argv[2], document_column="tweet_text")


topics, probabilities = topic_model.transform(vectoriser.fit_transform(documents))

cull_topics(
    topic_model,
    documents,
    TARGET_MAX=10
)

save(documents, topic_model, topics, sys.argv[2].replace(".txt", ""))
