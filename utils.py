import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import math
import pickle
import os
from collections import OrderedDict

def clones (module, N) :
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class HighWay (nn.Module) :
	"""
	Output = layerNorm(x + subLayer(x))

	This is a bit different from the QANet
	paper but this is the original formula
	in Attention Is All You Need
	"""
	def __init__ (self, size, dropout) :
		super(HighWay, self).__init__()
		self.norm = nn.LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward (self, x, subLayer) :
		return self.norm(x + self.dropout(subLayer(x)))

class PositionalEncoding (nn.Module) :
	"""
	In a way, for each position, the 
	positional encoding assigns a d-bit
	number which represents that position.
	d is the number of dimensions for that 
	position word.
	Just that this d-bit number is in
	continuous domain.
	"""

	def __init__ (self, dim, dropout, max_len=5000) :
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		pe = torch.zeros(max_len, dim)
		position = torch.arange(0, max_len).unsqueeze(1)
		freq = math.log(10000) / dim
		div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * freq))
		pe[:, 0::2] = torch.sin(position.float() * div_term)
		pe[:, 1::2] = torch.cos(position.float() * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)
		self.dim = dim

	def forward(self, x) :
		x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
		return self.dropout(x)


class PositionWiseFFN(nn.Module):

	def __init__(self, dim, h, dropout=0.1):
		super(PositionWiseFFN, self).__init__()
		self.w1 = nn.Linear(dim, h)
		self.w2 = nn.Linear(h, dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		return self.w2(self.dropout(F.relu(self.w1(x))))


class Pointer(nn.Module) :

	def __init__ (self, dm) :
		super(Pointer, self).__init__()
		self.w = nn.Parameter(torch.zeros((1, dm)))
		
	def forward(self, x) :
		return (self.w @ x).reshape(x.shape[0],x.shape[2])


class Glove (nn.Module) : 

	def __init__ (self, dim) :
		super(Glove, self).__init__()
		self.wordTable = loadGlove(dim)
		self.emb = nn.Embedding(len(self.wordTable), dim)
		weights = np.array(list(self.wordTable.values()))
		self.emb.weight.data.copy_(torch.from_numpy(weights))
		self.emb.weight.requires_grad = False

	def forward(self, x) : 
		return self.emb(x)

def loadGlove(dim) :
	word2vec = {}
	path = "data/glove/glove.6B." + str(dim) + 'd'
	if os.path.exists(path + '.cache') : 
		with open(path + '.cache', 'rb') as cache_file :
			word2vec = pickle.load(cache_file)

	else :
		with open(path + '.txt') as f :
			for line in f :
				l = line.split()
				word2vec[l[0]] = [float(x) for x in l[1:]]

		with open(path + '.cache', 'wb') as cache_file :
			pickle.dump(word2vec, cache_file)

	word2vec = OrderedDict(word2vec)
	return word2vec

def loss (y, idx) :
	l = torch.zeros(1)
	b = torch.ones(1) * y.shape[0]
	for i, r in enumerate(idx) : 
		l = l + torch.log(y[i, 0, r[0]]) + torch.log(y[i, 1, r[1]])
	return -l/b





