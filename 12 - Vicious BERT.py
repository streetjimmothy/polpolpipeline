import argparse
import math
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tqdm import tqdm


def main():
	parser = argparse.ArgumentParser(description="Run BERT predictions on a CSV or TXT file with batching and progress bar")
	parser.add_argument('-i', '--input-file', help='Input CSV/TXT file')
	parser.add_argument('-c', '--column-name', default='text', help='Column name containing text (default: text)')
	parser.add_argument('--batch-size', type=int, default=32, help='Batch size for tokenization/model (default: 32)')
	args = parser.parse_args()

	model_path = "model/fold_0/checkpoint-11"
	input_file = args.input_file
	column_name = args.column_name
	batch_size = args.batch_size
	output_csv = input_file.replace('.csv', '-vb.csv')

	# Device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Load model and tokenizer
	tokenizer = BertTokenizer.from_pretrained(model_path)
	model = BertForSequenceClassification.from_pretrained(model_path)
	model.to(device)
	model.eval()

	# Load data
	if input_file.endswith('.txt'):
		with open(input_file, 'r', encoding='utf-8') as f:
			texts = [line.strip() for line in f if line.strip()]
		df = pd.DataFrame({column_name: texts})
	else:
		df = pd.read_csv(input_file)
		if column_name not in df.columns:
			print(f"Input CSV must have a '{column_name}' column.")
			return

	texts = df[column_name].astype(str).tolist()
	total = len(texts)
	results = []

	# Predict in batches with progress bar
	with torch.no_grad():
		pbar = tqdm(total=total, desc='Predicting', unit='examples')
		for start in range(0, total, batch_size):
			end = min(start + batch_size, total)
			batch_texts = texts[start:end]
			inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=256)
			inputs = {k: v.to(device) for k, v in inputs.items()}
			outputs = model(**inputs)
			logits = outputs.logits
			preds = torch.argmax(logits, dim=1).cpu().tolist()
			results.extend(preds)
			pbar.update(len(batch_texts))
		pbar.close()

	df['bert_prediction'] = results
	df.to_csv(output_csv, index=False)
	print(f"Results saved to {output_csv}")


if __name__ == "__main__":
	main()
