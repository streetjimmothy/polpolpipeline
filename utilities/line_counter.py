import sys

def count_lines(filename):
	with open(filename, 'rb') as f:
		return sum(1 for line in f)

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print(f"Usage: {sys.argv[0]} <filename>")
	else:
		print(count_lines(sys.argv[1]))