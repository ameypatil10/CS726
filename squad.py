import ujson as json
import pandas as pd
import string
import re
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import utils

class SQuAD(torch.utils.data.Dataset):

	def __init__(self, jsonFile):
		self.dataFrame = pd.read_json(jsonFile)

	def __len__(self):
		return len(self.dataFrame)

	def __getitem__(self, idx):
		sample = {}
		sample['context'] = self.dataFrame.iloc[idx, 0]
		sample['question'] = self.dataFrame.iloc[idx, 1][0]
		sample['answer'] = self.dataFrame.iloc[idx, 1][1]
		return sample


def range(answer, context) :
	if len(answer) > len(context) :
		raise Exception("wrong")
	i = 0
	j = 0
	start = -1
	end = -1
	while i < len(context) : 
		if j >= len(answer) :
			return (i - j, i - 1)

		c = context[i]
		a = answer[j]
		
		if c == a:
			i += 1
			j += 1
		else :
			j = 0
			i += 1
	if j >= len(answer) :
		return (i - j, i - 1)
	else :
		return -1


def preprocess(path, dim) :

	with open((path + ".json"), 'r') as f :
		data = json.load(f)
	words = utils.loadGlove(dim)
	wordsList = list(words)
	unk = wordsList.index('unk')

	def word2idx (x) : 
		if x in words :
			return wordsList.index(x)
		else :
			return unk

	l = []
	for article in data['data'] : 
		for para in article['paragraphs'] :
			context = re.findall(r"[\w']+", para['context'])
			context = list(filter(None, context))
			context2idx = list(map(word2idx, context))
			for qa in para['qas'] : 
				if not qa['is_impossible'] :
					qc = {}
					ans = re.findall(r"[\w']+", qa['answers'][0]['text'])
					ans = list(filter(None, ans))
					ques = re.findall(r"[\w']+", qa['question'])
					ques = list(filter(None, ques))
					ques = list(map(word2idx, ques))
					r = range(ans, context)
					if r == -1 :
						continue
					qc['context'] = context2idx
					qc['qas'] = (ques, r)
					l.append(qc)

	outputFile = path + ".pre" + ".json"
	with open(outputFile, 'w+') as f : 
		json.dump(l,f)

def main () :
	preprocess('data/squad/dev-v2.0', 50)
	preprocess('data/squad/train-v2.0', 50)

if __name__ == "__main__" : 
	main()
