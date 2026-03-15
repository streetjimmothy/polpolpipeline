"""
08a - BERTopic.py
Spanish-tweet topic-modelling pipeline.

Embedding backbone : pysentimiento/robertuito-base-uncased  (RoBERTuito)
Topic model        : BERTopic
Acceleration       : cuML UMAP + HDBSCAN when available, otherwise CPU fallback
Caching            : embeddings are saved to a .npy file so re-runs skip encoding
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import argparse
import hashlib
import json
import numpy as np
from tqdm import tqdm
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from pathlib import Path
import utilities as utils

# ---------------------------------------------------------------------------
# GPU-accelerated dimensionality reduction / clustering (cuML).
# Falls back gracefully to CPU equivalents when cuML is unavailable.
# ---------------------------------------------------------------------------
try:
	from cuml.manifold import UMAP
	from cuml.cluster import HDBSCAN
	_CUML_AVAILABLE = True
except ImportError:
	from umap import UMAP
	from hdbscan import HDBSCAN
	_CUML_AVAILABLE = False
	print("cuML not available — using CPU UMAP/HDBSCAN.")

# ---------------------------------------------------------------------------
# RoBERTuito — Spanish RoBERTa pre-trained by PySentimiento.
# The model is loaded as a generic HuggingFace encoder via SentenceTransformer;
# sentence-transformers ≥ 2.2 handles models without a sentence-transformers
# config by applying mean-pooling automatically.
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "pysentimiento/robertuito-base-uncased"

# ---------------------------------------------------------------------------
# Spanish stopwords for the CountVectorizer used in c-TF-IDF.
# Covers core function words, common contractions, and Twitter noise.
# ---------------------------------------------------------------------------
SPANISH_STOPWORDS = [
	# articles & determiners
	"el", "la", "los", "las", "un", "una", "unos", "unas",
	# prepositions
	"a", "al", "ante", "bajo", "con", "de", "del", "desde", "en", "entre",
	"hacia", "hasta", "para", "por", "según", "sin", "sobre", "tras",
	# conjunctions
	"e", "ni", "o", "u", "pero", "mas", "sino", "aunque", "porque",
	"como", "cuando", "donde", "mientras", "que", "si", "ya",
	# pronouns
	"yo", "tú", "él", "ella", "ello", "nosotros", "nosotras",
	"vosotros", "vosotras", "ellos", "ellas",
	"me", "te", "se", "nos", "os", "le", "les",
	"mi", "tu", "su", "nuestro", "nuestra", "vuestro", "vuestra",
	"este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
	"aquel", "aquella", "aquellos", "aquellas",
	"lo", "le", "les", "les",
	# common verbs (highly frequent, low information)
	"es", "son", "era", "eran", "fue", "fueron", "ser", "sido",
	"ha", "han", "hay", "haber", "he", "has", "hemos", "habéis",
	"tiene", "tienen", "tener", "tengo", "tenemos",
	"hace", "hacen", "hacer", "hizo", "va", "van", "ir",
	"puede", "pueden", "poder", "quiero", "quieren", "querer",
	"soy", "somos", "estoy", "está", "están", "estar", "estaba",
	# adverbs
	"no", "sí", "más", "también", "muy", "bien", "mal", "así",
	"solo", "sólo", "aquí", "ahí", "allí", "ahora", "antes",
	"después", "siempre", "nunca", "todo", "todos", "toda",
	"nada", "algo", "alguien", "nadie",
	# Twitter-specific noise
	"rt", "via", "http", "https", "amp",
]


# ===========================================================================
# Embedding cache helpers
# ===========================================================================

def _docs_fingerprint(documents: list[str]) -> str:
	"""SHA-256 over all document bytes — used to detect corpus changes."""
	h = hashlib.sha256()
	for doc in documents:
		h.update(doc.encode("utf-8", errors="replace"))
	return h.hexdigest()


def load_or_compute_embeddings(
	documents: list[str],
	cache_path: str | Path,
	batch_size: int = 256,
	device: str = "cuda",
) -> np.ndarray:
	"""
	Return sentence embeddings for *documents*, using RoBERTuito.

	If *cache_path* exists and its companion ``.meta.json`` records the same
	document fingerprint and model name, the cached ``.npy`` array is returned
	directly — no GPU encoding needed.  Otherwise the model encodes the corpus
	and writes both files for future runs.

	Parameters
	----------
	documents  : list of raw tweet strings (already cleaned / de-noised).
	cache_path : path to the ``.npy`` file (meta JSON placed alongside it).
	batch_size : sentences per forward pass (tune for GPU VRAM).
	device     : ``"cuda"`` or ``"cpu"``.
	"""
	cache_path = Path(cache_path)
	meta_path = cache_path.with_suffix("").with_suffix(".meta.json")
	# ↑ strips the last extension then appends .meta.json, so
	#   "out/embeddings.npy" → "out/embeddings.meta.json"

	doc_hash = _docs_fingerprint(documents)

	# --- try cache ---
	if cache_path.exists() and meta_path.exists():
		try:
			with open(meta_path, encoding="utf-8") as fh:
				meta = json.load(fh)
			if meta.get("doc_hash") == doc_hash and meta.get("model") == EMBEDDING_MODEL_NAME:
				print(f"[cache] Loading embeddings from {cache_path}  ({len(documents):,} docs, model match ✓)")
				return np.load(str(cache_path))
			else:
				reason = "model changed" if meta.get("model") != EMBEDDING_MODEL_NAME else "corpus changed"
				print(f"[cache] Stale cache ({reason}) — re-encoding.")
		except Exception as exc:
			print(f"[cache] Read error ({exc}) — re-encoding.")

	# --- encode ---
	print(f"[embed] Loading {EMBEDDING_MODEL_NAME} on {device} ...")
	embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

	print(f"[embed] Encoding {len(documents):,} documents (batch_size={batch_size}) ...")
	all_embeddings: list[np.ndarray] = []
	for i in tqdm(range(0, len(documents), batch_size), desc="Embedding batches"):
		batch = documents[i : i + batch_size]
		emb = embedder.encode(batch, show_progress_bar=False, convert_to_numpy=True)
		all_embeddings.append(emb)
	embeddings = np.vstack(all_embeddings)

	# --- save cache ---
	cache_path.parent.mkdir(parents=True, exist_ok=True)
	np.save(str(cache_path), embeddings)
	meta = {
		"model": EMBEDDING_MODEL_NAME,
		"doc_hash": doc_hash,
		"n_docs": len(documents),
		"embedding_dim": embeddings.shape[1],
	}
	with open(meta_path, "w", encoding="utf-8") as fh:
		json.dump(meta, fh, indent=2)
	print(f"[cache] Saved embeddings → {cache_path}  (shape {embeddings.shape})")

	return embeddings


# ===========================================================================
# BERTopic pipeline
# ===========================================================================

def _build_vectorizer(
	extra_stopwords: list[str] | None = None,
	min_df: int = 5,
	max_df: float = 0.85,
) -> CountVectorizer:
	"""CountVectorizer with Spanish stopwords for c-TF-IDF."""
	stopwords = list(SPANISH_STOPWORDS)
	if extra_stopwords:
		stopwords = list(set(stopwords) | set(extra_stopwords))
	return CountVectorizer(
		ngram_range=(1, 2),
		stop_words=stopwords,
		min_df=min_df,
		max_df=max_df,
	)


def run_bertopic(
	documents: list[str],
	cache_path: str | Path | None = None,
	vectoriser: CountVectorizer | None = None,
	extra_stopwords: list[str] | None = None,
	min_cluster_size: int = 50,
	verbose: bool = True,
	batch_size: int = 256,
	device: str = "cuda",
) -> tuple:
	"""
	Encode with RoBERTuito then fit BERTopic.

	Returns
	-------
	topic_model : fitted BERTopic instance
	topics      : list[int] — per-document topic assignment (-1 = outlier)
	embeddings  : np.ndarray of shape (n_docs, embedding_dim)
	"""
	print(f"[bertopic] Pipeline start — {len(documents):,} documents")

	# ------------------------------------------------------------------
	# 1. Embeddings (cached)
	# ------------------------------------------------------------------
	if cache_path is None:
		cache_path = Path("bertopic_robertuito_embeddings.npy")
	embeddings = load_or_compute_embeddings(
		documents,
		cache_path=cache_path,
		batch_size=batch_size,
		device=device,
	)

	# ------------------------------------------------------------------
	# 2. Sub-models
	# ------------------------------------------------------------------
	ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
	representation_model = MaximalMarginalRelevance(diversity=0.3)

	if _CUML_AVAILABLE:
		umap_model = UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=42)
		hdbscan_model = HDBSCAN(
			min_cluster_size=int(min_cluster_size),
			metric="euclidean",
			cluster_selection_method="eom",
			prediction_data=False,
		)
	else:
		umap_model = UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=42, low_memory=True)
		hdbscan_model = HDBSCAN(
			min_cluster_size=int(min_cluster_size),
			metric="euclidean",
			cluster_selection_method="eom",
			prediction_data=False,
		)

	vec_model = vectoriser if vectoriser is not None else _build_vectorizer(extra_stopwords)

	# ------------------------------------------------------------------
	# 3. BERTopic
	# ------------------------------------------------------------------
	topic_model = BERTopic(
		embedding_model=None,        # embeddings are pre-computed
		umap_model=umap_model,
		hdbscan_model=hdbscan_model,
		vectorizer_model=vec_model,
		ctfidf_model=ctfidf_model,
		representation_model=representation_model,
		verbose=verbose,
		calculate_probabilities=False,
	)

	print("[bertopic] Fitting BERTopic on pre-computed embeddings ...")
	topics, _ = topic_model.fit_transform(pd.Series(documents), embeddings)

	initial_count = len({t for t in topics if t != -1})
	print(f"[bertopic] Initial non-outlier topics: {initial_count}")

	return topic_model, topics, embeddings


def cull_topics(
	topic_model: BERTopic,
	documents: list[str],
	target_max: int = 50,
) -> None:
	"""Merge fine-grained topics down to *target_max* using BERTopic's built-in reducer."""
	print(f"[bertopic] Reducing to ≤{target_max} topics ...")
	topic_model.reduce_topics(documents, nr_topics=target_max)
	final_count = len({t for t in topic_model.topics_ if t != -1})
	print(f"[bertopic] Topics after reduction: {final_count}  (total incl. outlier: {len(set(topic_model.topics_))})")


def save(
	documents: list[str],
	topic_model: BERTopic,
	topics: list[int],
	output_file_base: str = "tweets_with_topics",
) -> None:
	"""Persist per-document topic assignments and aggregate topic info."""
	print("[save] Writing per-document topic assignments ...")
	output_path = Path(f"{output_file_base}_with_topics.csv")
	output_path.parent.mkdir(parents=True, exist_ok=True)

	with open(output_path, "w", encoding="utf-8", newline="") as fh:
		fh.write("document_index,document,topic\n")
		for i, topic in tqdm(enumerate(topics), total=len(topics), desc="Writing rows"):
			# Escape any commas or newlines in the tweet text
			safe_doc = '"' + documents[i].replace('"', '""') + '"'
			fh.write(f"{i},{safe_doc},{topic}\n")

	print("[save] Writing topic_info CSV ...")
	topic_info = topic_model.get_topic_info()
	topic_info.to_csv(f"{output_file_base}-topic_info.csv", index=False)
	print("[save] Done.")


def plot(
	topic_model: BERTopic,
	topics: list[int],
	output_file_base: str = "tweets_with_topics",
) -> None:
	"""Generate interactive Plotly visualisations."""
	final_count = len({t for t in topics if t != -1})
	top_n = min(50, final_count)
	print(f"[plot] Generating visualisations for {top_n} topics ...")

	topic_model.visualize_barchart(top_n_topics=top_n).write_html(
		f"{output_file_base}_topic-barchart.html"
	)
	topic_model.visualize_topics().write_html(
		f"{output_file_base}_topic-visualization.html"
	)
	topic_model.visualize_hierarchy().write_html(
		f"{output_file_base}_topic-hierarchy.html"
	)

	print(topic_model.get_topic_info().head(15).to_string())


# ===========================================================================
# Convenience entry-point (called from 08 - topic_modelling.py)
# ===========================================================================

def run(
	documents: list[str],
	output_file_base: str,
	cache_path: str | Path | None = None,
	vectoriser: CountVectorizer | None = None,
	extra_stopwords: list[str] | None = None,
	min_cluster_size: int = 50,
	max_topics: int = 50,
	batch_size: int = 256,
	device: str = "cuda",
	verbose: bool = True,
) -> None:
	"""
	Full pipeline:  embed → cluster → reduce → save → plot.

	Parameters
	----------
	documents         : list of cleaned tweet strings.
	output_file_base  : prefix for all output files (CSV, HTML).
	cache_path        : path to the ``.npy`` embedding cache
	                    (default: ``<output_file_base>_robertuito_embeddings.npy``).
	vectoriser        : optional pre-built CountVectorizer; if None a Spanish one
	                    is constructed automatically.
	extra_stopwords   : additional stopwords merged with the built-in Spanish list.
	min_cluster_size  : HDBSCAN minimum cluster size (tune for corpus size).
	max_topics        : target upper bound after topic reduction.
	batch_size        : encoding batch size (tune for GPU VRAM).
	device            : ``"cuda"`` or ``"cpu"``.
	verbose           : pass-through to BERTopic verbosity.
	"""
	if cache_path is None:
		cache_path = Path(f"{output_file_base}_robertuito_embeddings.npy")

	topic_model, topics, _embeddings = run_bertopic(
		documents,
		cache_path=cache_path,
		vectoriser=vectoriser,
		extra_stopwords=extra_stopwords,
		min_cluster_size=min_cluster_size,
		verbose=verbose,
		batch_size=batch_size,
		device=device,
	)
	cull_topics(topic_model, documents, target_max=max_topics)
	save(documents, topic_model, topics, output_file_base)
	plot(topic_model, topics, output_file_base)


# ===========================================================================
# CLI — standalone usage
# ===========================================================================

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description=(
			"BERTopic topic modelling for Spanish tweets using RoBERTuito embeddings.\n"
			"Embeddings are cached in a .npy file to skip re-encoding on subsequent runs."
		)
	)
	utils.create_input_args(parser, ext="-denoised.txt")
	utils.create_output_args(parser, suffix="_{topic-barchart|topic-info}.{html|csv}")

	parser.add_argument(
		"--document-column",
		default="tweet_text",
		help="Column name in CSV input that contains the tweet text (default: tweet_text).",
	)
	parser.add_argument(
		"--max-topics",
		type=int,
		default=50,
		help="Maximum topics after reduction (default: 50).",
	)
	parser.add_argument(
		"--min-cluster-size",
		type=int,
		default=50,
		help="HDBSCAN min_cluster_size — increase to get fewer, larger topics (default: 50).",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=256,
		help="Encoding batch size; lower if you hit GPU OOM (default: 256).",
	)
	parser.add_argument(
		"--device",
		default="cuda",
		choices=["cuda", "cpu"],
		help="Device for encoding (default: cuda).",
	)
	parser.add_argument(
		"--cache",
		default=None,
		help="Explicit path for the .npy embedding cache (optional).",
	)
	parser.add_argument(
		"--stop-words",
		default=None,
		help="Path to a plain-text file of additional stopwords (one per line).",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		default=True,
		help="Enable verbose BERTopic output.",
	)

	args = parser.parse_args()

	extra_sw: list[str] | None = None
	if args.stop_words:
		sw_path = Path(args.stop_words)
		extra_sw = [ln.strip() for ln in sw_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
		print(f"Loaded {len(extra_sw)} extra stopwords from {sw_path}")

	input_files = utils.parse_input_files_arg(args.input_file, ext="-denoised.txt")
	output_files = utils.parse_output_files_arg(args.output, input_files)

	for input_file, output_file in zip(input_files, output_files):
		in_path = Path(input_file)
		# Support both plain .txt (one tweet per line) and .csv
		if in_path.suffix.lower() == ".csv":
			df = pd.read_csv(in_path)
			docs = df[args.document_column].dropna().astype(str).tolist()
		else:
			docs = [ln for ln in in_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

		print(f"\n{'='*60}")
		print(f"Input  : {in_path}  ({len(docs):,} documents)")
		print(f"Output : {output_file}")
		print(f"{'='*60}\n")

		cache = Path(args.cache) if args.cache else None

		run(
			documents=docs,
			output_file_base=str(output_file),
			cache_path=cache,
			extra_stopwords=extra_sw,
			min_cluster_size=args.min_cluster_size,
			max_topics=args.max_topics,
			batch_size=args.batch_size,
			device=args.device,
			verbose=args.verbose,
		)
