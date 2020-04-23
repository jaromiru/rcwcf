from config import *

import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

#==============================
flatten = lambda z: [x for y in z for x in y]

#==============================
def roll(x, n):  
	return torch.cat((x[-n:], x[:-n]))

#==============================
def seg_mean_vec(x, x_len):
	tot_len = len(x_len)

	if x.shape[0] == 0:
		x_agg = torch.zeros((tot_len, config.BAG_SIZE), device=config.DEVICE)
	else:
		x_len_idx  	  = torch.LongTensor(x_len).to(config.DEVICE)
		x_len_nonzero = (x_len_idx != 0)		# to deal with empty inputs

		sum_indexes = torch.cumsum(x_len_idx, dim=0) - 1
		sum_batch = torch.cumsum(x, dim=0)[sum_indexes]
		sum_batch = sum_batch[x_len_nonzero]

		sum_sub = roll(sum_batch, 1)
		sum_sub[0] = 0

		x_len_float = x_len_idx.view(-1, 1).type(torch.float32)[x_len_nonzero]
		x_agg = (sum_batch - sum_sub) / x_len_float			

		x_agg_zero = torch.zeros((tot_len, config.BAG_SIZE), device=config.DEVICE)
		x_agg_zero[x_len_nonzero] = x_agg

		x_agg = x_agg_zero

	return x_agg

#==============================
def seg_mean_seq(x, x_len):
	tot_len = len(x_len)

	x = torch.split(x, x_len)
	x_agg = torch.zeros((tot_len, config.BAG_SIZE), device=config.DEVICE)
	for idx in range(tot_len):
		if x[idx].shape[0] != 0:
			x_agg[idx] = torch.mean(x[idx], dim=0)

	return x_agg

class CustomBatchNorm(torch.nn.BatchNorm1d):
	def forward(self, x):
		if len(x) > 1:
			x = super().forward(x)

		else:	# use running stats instead
			training_flag = self.training 
			self.training = False
			x = super().forward(x)
			self.training = training_flag

		return x

#==============================
class BagInput(torch.nn.Module):
	def __init__(self, meta, act_f, agg_f):
		super().__init__()

		feat_len = meta.feat_idx[-1][1]

		self.f = torch.nn.Linear(feat_len, config.BAG_SIZE) 
		self.norm = CustomBatchNorm(config.BAG_SIZE)

		self.act_f = act_f
		self.agg_f = agg_f

		self.meta = meta

		self.sub_bags = []
		for fid, bag in meta.bags:
			module = BagInput(bag, F.relu, seg_mean_vec)
			self.sub_bags.append(module)
			self.add_module(f"bag_{fid}", module)

	''' Return bag values '''
	def forward(self, bag_data, bags_len):
		x = torch.stack( [smp.feats for smp in bag_data] )

		# get sub-bags
		for bid, (fid, bag) in enumerate(self.meta.bags):
			sub_bag_data, sub_bags_len = zip( *[(smp.bags[bid], len(smp.bags[bid])) for smp in bag_data] ) 
			sub_bag_data = flatten(sub_bag_data)

			if sub_bag_data:
				sub_bag_cmp = self.sub_bags[bid](sub_bag_data, sub_bags_len) # compute result for each sample 

				fpos = self.meta.feat_idx[fid]
				x[:, fpos[0]:fpos[1]] = sub_bag_cmp # insert computed data into the current features

		x = self.f(x)
		if self.act_f is not None: x = self.act_f(x)
		if self.agg_f is not None: x = self.agg_f(x, bags_len)

		x = self.norm(x)

		return x

#==============================
class Net(torch.nn.Module):
	def __init__(self, meta):
		super().__init__()

		self.model_in  = BagInput(meta, F.relu, None)
		self.model_cls = torch.nn.Linear(config.BAG_SIZE, config.CLASSES)		# classifying actions

		self.opt = torch.optim.Adam(self.parameters(), lr=config.OPT_LR, weight_decay=config.OPT_L2)
		self.loss = torch.nn.CrossEntropyLoss()

	def forward(self, batch):
		x = self.model_in(batch, None)
		o_cls = self.model_cls(x)

		return o_cls

	def predict(self, batch):
		res = self(batch)
		_, res = torch.max(res, dim=1)

		return res

	def train(self, batch_x, batch_y):
		y_ = self(batch_x)
		y  = torch.LongTensor(batch_y)

		# print(batch_x[:2])
		# print(batch_y[:2])
		# print(y_[:2].argmax(dim=1))
		# input()

		loss = self.loss(y_, y)

		self.opt.zero_grad()
		loss.backward()
		self.opt.step()

		return loss.item()

	def save(self, file='model'):
		torch.save(self.state_dict(), file)

	def load(self, file='model'):	    
	    self.load_state_dict(torch.load(file, map_location=config.DEVICE))

	def set_lr(self, lr):
		for param_group in self.opt.param_groups:
			param_group['lr'] = lr