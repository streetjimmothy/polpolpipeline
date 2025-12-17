import numpy as np
import json
import matplotlib.pyplot as plt
import csv
import os
import argparse

community_data = {}
averages = {}

plots_to_generate = {
	"Summary Variables": ['LIWC22.Analytic', 'LIWC22.Clout', 'LIWC22.Authentic', 'LIWC22.Tone'],
	"Dictionary Statistics": ['LIWC22.WC', 'LIWC22.WPS', 'LIWC22.BigWords', 'LIWC22.Dic'],
	"Linguistic Dimensions": ['LIWC22.Linguistic', 'LIWC22.function', 'LIWC22.pronoun', 'LIWC22.ppron', 'LIWC22.i', 'LIWC22.we', 'LIWC22.you', 'LIWC22.shehe', 'LIWC22.they', 'LIWC22.ipron', 'LIWC22.det', 'LIWC22.article', 'LIWC22.number', 'LIWC22.prep', 'LIWC22.auxverb', 'LIWC22.adverb', 'LIWC22.conj', 'LIWC22.negate', 'LIWC22.verb', 'LIWC22.adj', 'LIWC22.quantity'],
	"Drives": ['LIWC22.Drives', 'LIWC22.affiliation', 'LIWC22.achieve', 'LIWC22.power'],
	"Cognition": ['LIWC22.allnone', 'LIWC22.cogproc', 'LIWC22.insight', 'LIWC22.cause', 'LIWC22.discrep', 'LIWC22.tentat', 'LIWC22.certitude', 'LIWC22.differ', 'LIWC22.memory'],
	"Affect": ['LIWC22.Affect', 'LIWC22.tone_pos', 'LIWC22.tone_neg', 'LIWC22.emotion', 'LIWC22.emo_pos', 'LIWC22.emo_neg', 'LIWC22.emo_anx', 'LIWC22.emo_anger', 'LIWC22.emo_sad', 'LIWC22.swear'],
	"Social": ['LIWC22.Social', 'LIWC22.socbehav', 'LIWC22.prosocial', 'LIWC22.polite', 'LIWC22.conflict', 'LIWC22.moral', 'LIWC22.comm'],
	"Social References": ['LIWC22.socrefs', 'LIWC22.family', 'LIWC22.friend', 'LIWC22.female', 'LIWC22.male'],
	"Culture": ['LIWC22.Culture', 'LIWC22.politic', 'LIWC22.ethnicity', 'LIWC22.tech'],
	"Lifestyle": ['LIWC22.Lifestyle', 'LIWC22.leisure', 'LIWC22.home', 'LIWC22.work', 'LIWC22.money', 'LIWC22.relig'],
	"Physical": ['LIWC22.Physical', 'LIWC22.health', 'LIWC22.illness', 'LIWC22.wellness', 'LIWC22.mental', 'LIWC22.substances', 'LIWC22.sexual', 'LIWC22.food', 'LIWC22.death'],
	"Needs": ['LIWC22.need', 'LIWC22.want', 'LIWC22.acquire', 'LIWC22.lack', 'LIWC22.fulfill', 'LIWC22.fatigue'],
	"Motivation": ['LIWC22.reward', 'LIWC22.risk', 'LIWC22.curiosity', 'LIWC22.allure'],
	"Perception": ['LIWC22.Perception', 'LIWC22.attention', 'LIWC22.motion', 'LIWC22.space', 'LIWC22.visual', 'LIWC22.auditory', 'LIWC22.feeling', 'LIWC22.time', 'LIWC22.focuspast', 'LIWC22.focuspresent', 'LIWC22.focusfuture'],
	"Conversation": ['LIWC22.Conversation', 'LIWC22.netspeak', 'LIWC22.assent', 'LIWC22.nonflu', 'LIWC22.filler'],
	"Punctuation": ['LIWC22.AllPunc', 'LIWC22.Period', 'LIWC22.Comma', 'LIWC22.QMark', 'LIWC22.Exclam', 'LIWC22.Apostro', 'LIWC22.OtherP', 'LIWC22.Emoji']
}


std_devs = {
	'Analytic': 27.48, 'Clout': 28.36, 'Authentic': 25.58, 'Tone': 26.36,
	'WC': 2858.2, 'WPS': 100.62, 'BigWords': 5.63, 'Dic': 7.54,
	'Linguistic': 9.59,
	'function': 8.39,
	'pronoun': 5.17, 'ppron': 4.16, 'i': 3.57, 'we': 1.02, 'you': 1.62, 'shehe': 0.71, 'they': 0.39, 'ipron': 1.58,
	'det': 2.63, 'article': 1.73, 'number': 2.00,
	'prep': 2.87, 'auxverb': 2.51, 'adverb': 1.83, 'conj': 1.44, 'negate': 1.02, 'verb': 3.80, 'adj': 2.11, 'quantity': 1.13,
	'Drives': 2.32,
	'affiliation': 1.47, 'achieve': 1.19, 'power': 1.15,
	'Cognition': 3.24,
	'allnone': 0.83,
	'cogproc': 2.84, 'insight': 0.84, 'cause': 0.60, 'discrep': 0.79, 'tentat': 0.69, 'certitude': 0.41, 'differ': 1.05, 'memory': 0.12,
	'Affect': 4.48,
	'tone_pos': 4.52, 'tone_neg': 1.07, 'emotion': 2.13, 'emo_pos': 1.96, 'emo_neg': 0.56, 'emo_anx': 0.11, 'emo_anger': 0.20, 'emo_sad': 0.23, 'swear': 1.42,
	'Social': 4.83,
	'socbehav': 3.99, 'prosocial': 0.97, 'polite': 3.74, 'conflict': 0.25, 'moral': 0.40, 'comm': 0.97,
	'socrefs': 2.66,
	'family': 0.87, 'friend': 0.29, 'female': 0.63, 'male': 0.96,
	'Culture': 1.25,
	'politic': 0.92, 'ethnicity': 0.32, 'tech': 0.69,
	'Lifestyle': 2.60,
	'leisure': 0.78, 'home': 0.43, 'work': 2.30, 'money': 0.74, 'relig': 1.16,
	'Physical': 1.23,
	'health': 0.58, 'illness': 0.21, 'wellness': 0.10, 'mental': 0.08, 'substances': 0.23, 'sexual': 0.23, 'food': 0.68, 'death': 0.26,
	'need': 0.32, 'want': 0.39, 'acquire': 0.49, 'lack': 0.11, 'fulfill': 0.29, 'fatigue': 0.12,
	'reward': 0.54, 'risk': 0.25, 'curiosity': 0.32, 'allure': 2.25,
	'Perception': 2.32,
	'attention': 0.59, 'motion': 0.73, 'space': 1.89, 'visual': 0.64, 'auditory': 0.51, 'feeling': 0.31,
	'time': 3.68, 'focuspast': 1.47, 'focuspresent': 1.79, 'focusfuture': 0.69,
	'Conversation': 2.29,
	'netspeak': 2.08, 'assent': 0.46, 'nonflu': 0.32, 'filler': 0.18,
	'AllPunc': 10.97, 'Period': 5.04, 'Comma': 1.51, 'QMark ': 0.96, 'Exclam': 4.63, 'Apostro': 1.91, 'OtherP': 7.10
}

duplicate_values = ['WC', 'WPS', 'BigWords', 'AllPunc', 'Period', 'Comma', 'QMark', 'Exclam', 'Apostro', 'OtherP', 'Emoji']


def import_liwc_csv(dir, community_mappings):
	for csv_file in os.listdir(dir):
		if not csv_file.endswith('.csv'):
			continue
		# Assumes filename format: [countrycode]_[prunetype]_[communityinfo]_tweets_[dicname]-dictionary.csv
		community = csv_file.split('_')[2][0]
		community_name = community_mappings[community]
		if community_name not in community_data:
			community_data[community_name] = {}
		dictionary = csv_file.split('_')[-1].replace('-dictionary', '').replace('.csv', '')
		if dictionary not in community_data[community_name]:
			community_data[community_name][dictionary] = {}

		with open(os.path.join(dir, csv_file), newline='', encoding='utf-8') as f:
			reader = csv.DictReader(f)
			for row in reader:
				for header in reader.fieldnames:
					value = row[header]
					if header == 'Segment':  # we never want this
						continue
					if header in duplicate_values and dictionary != 'LIWC22':  # expanded dictionaries seem to duplicate these values from the main dictionary
						continue
					if header == 'Dic' and dictionary != 'LIWC22':  # Dic is uniquie to each dictionary, so the label needs to icnldue the dictionary name
						header = f'Dic_{dictionary}'
					try:
						value = float(value)
					except ValueError:
						continue  # Skip non-numeric values
					community_data[community_name][dictionary][header] = value


def calculate_averages():

	counts = {}

	for community, data in community_data.items():
		for dictionary, entry in data.items():
			for label, value in entry.items():
				unique_name = f"{dictionary}.{label}"
				if unique_name not in averages:
					averages[unique_name] = 0
					counts[unique_name] = 0
				averages[unique_name] += value
				counts[unique_name] += 1

	for label in averages:
		averages[label] /= counts[label]


def plot_variance(
	labels, community_colour_mappings,
	title=None, filename=None, include_steddev=False,
):
	_labels = {}
	for label in labels:
		_labels[label] = {"dic": label.split('.')[0], "label": label.split('.')[1]}

	baseline = [1] * len(labels)

	label_variance = {}
	std_devs_pos = []
	std_devs_neg = []
	for label in labels:
		_dic = _labels[label]['dic']
		_label = _labels[label]['label']
		for community, data in community_data.items():
			if _dic not in data:
				continue
			if community not in label_variance:
				label_variance[community] = []
			var = (data[_dic][_label] / averages[label]) if averages[label] != 0 else 1
			label_variance[community].append(var)
		if include_steddev:
			if _label not in std_devs:
				std_devs_pos.append(0)
				std_devs_neg.append(0)
			else:
				std_devs_pos.append(std_devs[_label] / averages[label])
				std_devs_neg.append(-std_devs[_label] / averages[label])

	offset = np.arange(len(label_variance))
	offset = (offset - np.mean(offset)) / (2 * len(label_variance))

	# Plot

	# If all the dictionaries are the same, we can just plot the label names
	lbl_cnt = {}
	for label_info in _labels.values():
		if label_info['dic'] not in lbl_cnt:
			lbl_cnt[label_info['dic']] = 0
		lbl_cnt[label_info['dic']] += 1

	if len(lbl_cnt) == 1:
		labels = [label_info['label'] for label_info in _labels.values()]

	x = 0.5 + np.arange(len(labels))
	fig = plt.figure(
		figsize=(10, 4),
		dpi=160
	)
	ax = fig.add_subplot(111)
	ax.plot(x, baseline, '_', color='black', markersize=100 * len(labels))
	for idx, (community, variances) in enumerate(label_variance.items()):
		stem = ax.stem(
			x + offset[idx],
			variances,
			linefmt='-',
			basefmt=' ',
			bottom=1,
		)
		plt.setp(stem.markerline, color=community_colour_mappings[community])
	ax.set_ylabel('Variance from Average')
	ax.set_title(f'{title}')
	ax.set(
		xlim=(0, len(labels)),
		xticks=x,
		xticklabels=labels,
		ylim=(0, 2)
	)

	if include_steddev:
		ax.bar(x, std_devs_pos, color=('xkcd:slate', 0.8), label='Std Dev', zorder=5, bottom=1)
		ax.bar(x, std_devs_neg, color=('xkcd:slate', 0.8), label='Std Dev', zorder=5, bottom=1)

	if filename is not None:
		plt.savefig(filename, dpi=300, bbox_inches='tight')
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Plots a collection of LIWC statistics from CSV files.")
	parser.add_argument("-i", "--input-dir", required=True, help="Path to the directory of CSV files with the LIWC stats")
	parser.add_argument("--community-info", required=False, help="JSON file to map community numbers onto names and colours", default="community_names.json")
	parser.add_argument("-t", "--title", required=False, help="Title for the plots", default="LIWC Statistics")
	parser.add_argument("--std-dev", action='store_true', help="If set, include standard deviation bars on the plots")
	args = parser.parse_args()

	community_mappings = {}
	community_colour_mappings = {}
	with open(args.community_info, 'r', encoding='utf-8') as f:
		j = json.load(f)
		community_mappings = j["labels"]
		community_colour_mappings = j["colours"]

	import_liwc_csv(args.input_dir, community_mappings)
	calculate_averages()

	for title, values in plots_to_generate.items():
		plot_variance(
			values,
			community_colour_mappings,
			title=f'{args.title} - {title}',
			filename=os.path.join(args.input_dir, f'{args.title}_{title}_variance.png'),
			include_steddev=args.std_dev
		)
