from model import QANet
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from squad import SQuAD 
import utils
import csv

DEV_JSON = 'data/squad/dev-v2.0.pre.json'
TRAIN_JSON = 'data/squad/train-v2.0.pre.json'
UNK = 201534
LEARNING_RATE = 1e-4

def collate(batch) :
	n = 0
	m = 0
	b = len(batch)
	for sample in batch :
		n = max(len(sample['context']), n)
		m = max(len(sample['question']), m)
	context = torch.ones(b, n) * UNK
	question = torch.ones(b, m) * UNK
	answer = []
	for idx, sample in enumerate(batch) :
		n_ = len(sample['context'])
		m_ = len(sample['question'])
		context[idx, :n_] = torch.tensor(sample['context'])
		question[idx, :m_] = torch.tensor(sample['question'])
		answer.append(sample['answer'])

	return [context.long(), question.long(), answer]

def main() :
	qanet = QANet(50)
	init1 = filter(lambda p : p.requires_grad and p.dim() >= 2, qanet.parameters())
	init2 = filter(lambda p : p.requires_grad and p.dim() <= 2, qanet.parameters())
	# Parameter initialization
	for param in init1 :
		nn.init.xavier_uniform_(param)
	for param in init2 :
		nn.init.normal_(param)

	train = SQuAD(TRAIN_JSON)
	val = SQuAD(DEV_JSON)

	# trainSet = DataLoader(dataset=train, batch_size=4, shuffle=True, collate_fn=collate)
	valSet = DataLoader(dataset=val, batch_size=4, shuffle=True, collate_fn=collate)
	trainSet = DataLoader(dataset=train, batch_size=4, shuffle=True, collate_fn=collate)

	print('length of dataloader', len(trainSet))

	optimizer = torch.optim.Adam(qanet.parameters(), lr=LEARNING_RATE)
	loss_list = []
	for epoch in range(10):
		print('epoch ',epoch)
		for i, (c, q, a )in enumerate(trainSet):
			y_pred = qanet(c, q)
			loss = utils.loss(y_pred, a)
			loss_list.append(loss.item())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if i % 200 == 0:
				print('loss ', loss.item())
		with open('your_file.txt', 'w') as f:
			for item in loss_list:
				f.write("%s\n" % item)
			print('loss file written.')
		torch.save(qanet, 'qanet')
		print('model saved.')



if __name__ == "__main__" :
	main()