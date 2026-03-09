import argparse
from pathlib import Path

from tqdm import tqdm


def fix_csv_line(line: str) -> str:
	#remove all quotes first
	line = line.replace('"', '')

	# Find first and last comma positions
	first_comma = line.find(',')
	last_comma = line.rfind(',')

	# Only process if at least one comma exists
	if first_comma != -1:
		if first_comma == last_comma:
			# Only one comma - replace with both ," and ",
			return line[:first_comma] + ',"' + line[first_comma + 1 :].rstrip('\n') + '",\n'
		# Multiple commas - replace first and last
		return line[:first_comma] + ',"' + line[first_comma + 1 : last_comma] + '",' + line[last_comma + 1 :]

	# No commas - keep line as is
	return line


def main() -> None:
	parser = argparse.ArgumentParser(description="Fix BERTopic CSV lines by quoting the middle field.")
	parser.add_argument(
		"--input",
		"-i",
		required=True,
		help="Path to the input file to fix (will be modified in-place).",
	)
	args = parser.parse_args()

	input_path = Path(args.input)
	if not input_path.exists() or not input_path.is_file():
		raise SystemExit(f"Input file not found: {input_path}")

	with input_path.open('r', encoding='utf-8') as f:
		lines = f.readlines()

	processed_lines = []
	for line in tqdm(lines, desc=f"Processing {input_path.name}", unit="lines"):
		processed_lines.append(fix_csv_line(line))

	with input_path.open('w', encoding='utf-8') as f:
		f.writelines(processed_lines)


if __name__ == "__main__":
	main()