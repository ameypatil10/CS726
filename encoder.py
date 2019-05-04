import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import utils
from attention import Attn
from utils import HighWay
from utils import PositionalEncoding
from utils import PositionWiseFFN

class EncoderBlock(nn.Module) :

	def __init__ (self, dm, dropout=0.1) :
		super(EncoderBlock, self).__init__()
		self.pe = PositionalEncoding(dm, dropout)
		self.self_attn = Attn()
		self.ffn = PositionWiseFFN(dm, dm//2)
		self.dropout = dropout
		self.highways = utils.clones(HighWay(dm, dropout), 2)

	def forward(self, x) :
		x = self.pe(x)
		x = self.highways[-2](x, self.self_attn)
		x = self.highways[-1](x, self.ffn)
		return x

class EncoderStack(nn.Module) :

	# FIX the sizes here as well
	def __init__ (self, block, nBlocks) :
		super(EncoderStack, self).__init__()
		self.blocks = utils.clones(block, nBlocks)

	def forward(self, x) :
		for block in self.blocks :
			x = block(x)
		return x




