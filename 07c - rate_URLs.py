import csv

domains = {}

def process_file(in_file, domain_csv, out_file):
	with open(domain_csv, 'r', encoding='utf-8', errors='ignore') as f:
		csv_reader = csv.reader(f)
		csv_reader.__next__()  # skip header
		for row in csv_reader:
			domains[row[0]] = {"label":row[1], "type":row[6]}
	with open(in_file, 'r', encoding='utf-8', errors='ignore') as fin, open(out_file, 'w', encoding='utf-8', errors='ignore') as fout:
		csv_writer = csv.writer(fout)
		csv_writer.writerow(["tweet_text", "label", "type"])
		for line in fin:
			for domain in domains:
				if domain in line:
					csv_writer.writerow([line.strip(), domains[domain]["label"], domains[domain]["type"]])
					break


