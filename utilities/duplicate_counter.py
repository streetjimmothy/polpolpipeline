import sys
from collections import Counter


def count_duplicate_lines(filename):
	with open(filename, 'r', encoding='utf-8') as f:
		lines = f.readlines()
	counts = Counter(lines)
	duplicates = sum(1 for count in counts.values() if count > 1)
	print(f"Total lines: {len(lines)}")
	print(f"Lines with duplicates:{duplicates}")
	print(f"Total unique lines: {len(counts)}")
	print(f"Number of duplicate lines: {len(lines) - len(counts)}")


if __name__ == "__main__":
	if len(sys.argv) != 2:
		print(f"Usage: {sys.argv[0]} <filename>")
		sys.exit(1)
	count_duplicate_lines(sys.argv[1])
