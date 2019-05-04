import sys
import utils
from utils import PositionalEncoding
from utils import PositionWiseFFN
from utils import Pointer
from utils import Glove
from encoder import EncoderBlock
from encoder import EncoderStack
from attention import ContextQueryAttention
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import copy

class QANet(nn.Module) :

	def __init__ (self, dm, dropout=0.1) :
		super(QANet, self).__init__()		
		self.cEmb = Glove(dm)
		self.qEmb = Glove(dm)
		# Fill the ffn dims
		# SEEB stands for Stacked Embedding Encoder Blocks
		self.qSEEB = EncoderStack(EncoderBlock(dm),1)
		self.cSEEB = EncoderStack(EncoderBlock(dm),1)

		# Fill in coAttns' sizes
		self.coAttn = ContextQueryAttention(dm)

		# Again fill in dimensions
		# SMEB stands for Stacked Model Encoder Blocks
		# Encoder block has 4*dm because the coAttn
		# outputs a 4*dm output. 
		# Maybe we can try convolving to reduce dimension
		self.SMEB = EncoderStack(EncoderBlock(4*dm), 7)

		self.pointer1 = Pointer(8*dm)
		self.pointer2 = Pointer(8*dm)

	def forward (self, context, question) :
		"""
		Input to this model is a batch of (say) B contexts 
		and B questions. A particular context in the batch
		will have (say) n words and the corresponding 
		question will have m words. Obviously, n and m 
		will vary across the batch.
		"""
		context = self.cEmb(context)
		question = self.qEmb(question)
		aa = context.shape
		
		context = self.cSEEB(context).permute(0, 2, 1)
		question = self.qSEEB(question).permute(0, 2, 1)

		abAttn = self.coAttn(question, context).permute(0, 2, 1)

		m0 = self.SMEB(abAttn)
		m1 = self.SMEB(m0)
		m2 = self.SMEB(m1)

		startIn = torch.cat([m0, m1], dim=2).permute(0, 2, 1)
		endIn = torch.cat([m0, m2], dim=2).permute(0, 2, 1)

		p1 = F.softmax(self.pointer1(startIn), dim=1)
		p2 = F.softmax(self.pointer2(endIn), dim=1)

		return torch.stack([p1,p2], dim=1)
