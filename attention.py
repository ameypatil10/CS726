import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import utils


class Attn (nn.Module) :

	def __init__ (self, dropout=0.1) :
		super(Attn, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, query, key=None, value=None) :
		if key is None or value is None :
			key = query
			value = query
		dk = query.size(-1)
		scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dk)
		pAttn = self.dropout(F.softmax(scores, dim=-1))
		return torch.matmul(pAttn, value)

class ContextQueryAttention(nn.Module) :

	"""
	Simple Context-to-query attention 
	"""

	def __init__(self, dm, dropout=0.1) :
		super(ContextQueryAttention, self).__init__()
		self.dm = dm
		
		self.w1 = nn.Parameter(torch.zeros((1, dm)))
		self.w2 = nn.Parameter(torch.zeros((1, dm)))
		self.w3 = nn.Parameter(torch.zeros((1, dm)))

		self.dropout = nn.Dropout(p=dropout)

	def forward (self, q, c) :
		"""
		query = [B x dm x m]
		context = [B x dm x n]
		"""
		B, d, m = q.shape
		B, d, n = c.shape
		
		q_ = q.reshape((B, d, 1, m))
		c_ = c.reshape((B, d, n, 1))
		
		qc = (q_ * c_).permute(0, 2, 1, 3)
		q_ = q_.permute(0, 2, 1, 3)
		c_ = c_.permute(0, 2, 1, 3)

		out1 = (self.w1 @ q_).reshape((B, 1, m))
		out2 = (self.w2 @ c_).reshape((B, n, 1))
		out3 = (self.w3 @ qc).reshape((B, n, m))

		s = (out3 + out2) + out1
		# Row wise normalization
		sRow = F.softmax(s, dim=2)
		# Col wise normalization
		sCol = F.softmax(s, dim=1)
		
		a = (sRow @ q.permute(0, 2, 1))
		a = a.permute(0, 2, 1)

		b = (sRow @ sCol.permute(0, 2, 1) @ c.permute(0, 2, 1))
		b = b.permute(0, 2, 1)

		return torch.cat([c, a, c * a, c * b], dim=1)


