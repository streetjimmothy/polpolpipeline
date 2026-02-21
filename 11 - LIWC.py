import subprocess
import os
import argparse


def analyze_text_with_liwc(input_file, output_dir=None, liwc_dictionary_path="LIWC22", liwc_executable_path='liwc-22-cli'):
	"""
	Analyzes the given text using the LIWC tool.

	Parameters:
	- input_file (str): File with the text to be analyzed.
	- liwc_executable_path (str): Path to the LIWC executable. Defaults to liwc-22-cli (assumes LIWC is on PATH).
	- liwc_dictionary_path (str): Path to the LIWC dictionary file. A folder uses each dictionary in the folder.
	- output_file (str): Path to save the LIWC output. If None, a default name will be used.
	"""

	command = [
		liwc_executable_path,
		'--mode', 'wc',
		'--input', input_file,
		'-t', '24'
	]

	if liwc_dictionary_path is not None:
		if os.path.isdir(liwc_dictionary_path):
			for dict_file in os.listdir(liwc_dictionary_path):
				if dict_file.endswith('.dicx'):
					analyze_text_with_liwc(
						input_file,
						output_dir=output_dir,
						liwc_dictionary_path=os.path.join(liwc_dictionary_path, dict_file),
						liwc_executable_path=liwc_executable_path
					)
			return
		else:
			command += ['-d', liwc_dictionary_path]

	if output_dir is None:
		output_dir = os.getcwd()

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	output_file = os.path.join(
		output_dir,
		os.path.splitext(os.path.split(input_file)[1])[0]
		+ f'_{os.path.splitext(os.path.split(liwc_dictionary_path)[1])[0]}'
		+ '.csv')
	command += ['--output', output_file]

	# Run the LIWC tool via subprocess
	result = subprocess.run(command)

	if result.returncode != 0:
		# TODO: Better error handling. Throw?
		print("Error running LIWC.")
		print(result.stderr)
		print(result.stdout)
	else:
		print(f"LIWC analysis complete. Output saved to {output_dir}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="LIWC Text Analysis")
	parser.add_argument("-i", "--input_file", "--input_folder", required=True, help="Path to the input text file to be analyzed. If a folder, each file in the folder will be processed.")
	parser.add_argument("-r", "--recursive", action='store_true', help="If set and input_file is a directory, process files in subdirectories recursively.")
	parser.add_argument("-o", "--output_dir", required=False, help="Path to save the LIWC output (provide a direcotry, and the output is saved as the input filename with dict suffix).")
	parser.add_argument("-d", "--liwc_dictionary_path", default="LIWC22", help="Path to the LIWC dictionary file or directory containing multiple dictionaries. Default is LIWC default")
	parser.add_argument("--liwc_executable_path", default='liwc-22-cli', help="Path to the LIWC executable (default assumes it's on PATH)")

	args = parser.parse_args()

	input_files = []

	if os.path.isdir(args.input_file):
		if args.recursive:
			for root, dirs, files in os.walk(args.input_file):
				for file_name in files:
					if file_name.endswith('.txt'):
						input_files.append(os.path.join(root, file_name))
		else:
			for file_name in os.listdir(args.input_file):
				if file_name.endswith('.txt'):
					input_files.append(os.path.join(args.input_file, file_name))
	else:
		input_files.append(args.input_file)

	for input_file in input_files:
		analyze_text_with_liwc(
			input_file=input_file,
			output_dir=args.output_dir,
			liwc_dictionary_path=args.liwc_dictionary_path,
			liwc_executable_path=args.liwc_executable_path
		)
