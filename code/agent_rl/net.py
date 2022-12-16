from config import *

import numpy as np

import torch
from torch.nn import LeakyReLU, LayerNorm
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

#==============================
def seg_expand(x, x_len):
	x_exp = []
	for i, e in enumerate(x):
		x_exp.append( x[i].expand(x_len[i], -1) )

	x_exp = torch.cat(x_exp, dim=0)
	return x_exp

#==============================
def get_all_p(it, bag_meta, root=False):
	p = torch.softmax(it.p.flatten(), dim=0).view_as(it.p)
	p_ = p.clone()

	allit_p = []
	for it_idx, it_bags in enumerate(it.bags):
		for bid, (fid, mbag) in enumerate(bag_meta.bags):
			it_bag = it_bags[bid]

			if it_bag and len(it_bag.p) != 0:
				if root:
					fid += config.CLASSES 	# these indicate classification actions

				if p[it_idx, fid] != 0.0:
					bag_p = p[it_idx, fid] * get_all_p(it_bag, mbag)
					allit_p.append(bag_p)

				p_[it_idx, fid] = 0.0


	allit_p.append(p_)
	allit_p = torch.cat([x.flatten() for x in allit_p])

	return allit_p

def entropy(samples, bag_meta):
	all_H = torch.empty(len(samples))

	for i, s in enumerate(samples):
		smp_p = get_all_p(s, bag_meta, root=True)
		smp_H = -torch.sum(smp_p * torch.log(smp_p + 1e-5))
		all_H[i] = smp_H

	return all_H

#==============================
class Result_In:
	def __init__(self, x, x_raw, x_mask, x_len, x_bags):
		self.x = x
		self.x_raw = x_raw # without aggregation
		self.x_mask = x_mask
		self.x_len = x_len
		self.x_bags = x_bags

#==============================
class Result_Out:
	def __init__(self, x, x_len, x_bags):
		self.x = x
		self.x_len = x_len
		self.x_bags = x_bags

#==============================
class Result_Item:
	def __init__(self, p, bags):
		self.p = p
		self.bags = bags

	def __str__(self):
		return f"{{p: {self.p}, bags: {self.bags}}}\n"

	__repr__ = __str__

def split(a, seg):
	if isinstance(seg, (list, tuple)):
		s = 0
		for l in seg:
			yield(a[s:s+l])
			s += l

	elif isinstance(seg, int):
		for s in range(0, len(a), seg):
			yield(a[s:s+seg])

def make_struct(res):
	orig_len = res.x.shape[0]

	x_bags = [[None] * orig_len if b is None else make_struct(b) for b in res.x_bags]
	x_bags = list(zip(*x_bags))		# transpose[item1 -> (bag1, bag2, ...), item2 -> ...]

	if res.x_len is None: # root node
		x = torch.split(res.x, 1)
		x_bags_sgm = split(x_bags, 1)

	else: 
		x = torch.split(res.x, res.x_len)
		x_bags_sgm = split(x_bags, res.x_len)

	items = [Result_Item(e, next(x_bags_sgm)) for e in x]
	return items

# class CustomBatchNorm(torch.nn.BatchNorm1d):
# 	def forward(self, x):
# 		if len(x) > 1:
# 			x = super().forward(x)

# 		else:	# use running stats instead
# 			training_flag = self.training 
# 			self.training = False
# 			x = super().forward(x)
# 			self.training = training_flag

# 		return x

#==============================
class BagInput(torch.nn.Module):
	def __init__(self, meta, act_f, agg_f):
		super().__init__()

		feat_len = meta.feat_idx[-1][1]

		self.f = torch.nn.Linear(feat_len + len(meta.feat_idx), config.BAG_SIZE) # features + mask
		self.act_f = act_f
		self.agg_f = agg_f
		self.norm = LayerNorm(config.BAG_SIZE)

		self.meta = meta

		self.sub_bags = []
		for fid, bag in meta.bags:
			module = BagInput(bag, act_f, seg_mean_vec) # <- can be replace with seg_mean_vec
			self.sub_bags.append(module)
			self.add_module(f"bag_{fid}", module)

		self.flag_no_mask = False

	''' Return bag values '''
	def forward(self, bag_data, bags_len):
		x 	= torch.stack( [smp.feats for smp in bag_data] )	

		if self.flag_no_mask: # in pre-training, data comes without the mask
			x_m = torch.ones( len(bag_data), len(self.meta.feat_idx) )
		else:	# do it normally
			x_m = torch.stack( [smp.mask for smp in bag_data] )

		# get sub-bags
		sub_bags = []

		for bid, (fid, bag) in enumerate(self.meta.bags):	# for each bag

			sub_bag_data, sub_bags_len = zip( *[(smp.bags[bid], len(smp.bags[bid])) for smp in bag_data] ) 
			sub_bag_data = flatten(sub_bag_data)

			if not sub_bag_data:
				sub_bags.append(None)

			else:
				res = self.sub_bags[bid](sub_bag_data, sub_bags_len) # compute result for each sample 
				sub_bags.append(res)

				fpos = self.meta.feat_idx[fid]
				x[:, fpos[0]:fpos[1]] = res.x 	# insert computed data into the current features

		x = torch.cat( (x, x_m), dim=1)
		x = self.f(x)
		
		if self.act_f is not None: x = self.act_f(x)
		x_raw = x
		if self.agg_f is not None: x = self.agg_f(x, bags_len)

		x = self.norm(x)

		return Result_In(x, x_raw, x_m, bags_len, sub_bags)

	def set_no_mask(self, flag_no_mask):
		for m in self.sub_bags:
			m.set_no_mask(flag_no_mask)

		self.flag_no_mask = flag_no_mask

	# ''' Return bag values and without filling bag features '''
	# def forward(self, bag_data, bags_len):
	# 	x = torch.stack( [smp.feats for smp in bag_data] )	

	# 	# get sub-bags
	# 	for bid, (fid, bag) in enumerate(self.meta.bags):
	# 		sub_bag_data, sub_bags_len = zip( *[(smp.bags[bid], len(smp.bags[bid])) for smp in bag_data] ) 
	# 		sub_bag_data = flatten(sub_bag_data)

	# 		sub_bag_cmp = self.sub_bags[bid](sub_bag_data, sub_bags_len) # compute result for each sample 

	# 		fpos = self.meta.feat_idx[fid]
	# 		x[:, fpos[0]:fpos[1]] = sub_bag_cmp # insert computed data into the current features

	# 	# TODO mask!
	# 	x = self.f(x)
	# 	if self.act_f is not None: x = self.act_f(x)
	# 	if self.agg_f is not None: x = self.agg_f(x, bags_len)

	# 	return x

#==============================
class BagOutput(torch.nn.Module):
	def __init__(self, meta, root):
		super().__init__()

		feat_len = meta.feat_idx[-1][1]
		feat_cnt = len(meta.feat_idx)

		if root:
			self.f = torch.nn.Linear(config.BAG_SIZE, feat_cnt)
		else:
			self.f = torch.nn.Linear(2 * config.BAG_SIZE, feat_cnt)

		self.root = root
		self.meta = meta

		self.sub_bags = []
		for fid, bag in meta.bags:
			module = BagOutput(bag, root=False)
			self.sub_bags.append(module)
			self.add_module(f"bag_{fid}", module)

	def forward(self, x_0, bag):
		if not self.root:
			x_0 = seg_expand(x_0, bag.x_len)
			x = torch.cat( (x_0, bag.x_raw), dim=1 )
		else:
			x = x_0

		x = self.f(x)
		x[ bag.x_mask >= 1.0 ] = -np.inf

		# forward to sub-bags
		sub_bags = []
		for bid, (fid, meta_bag) in enumerate(self.meta.bags):
			sbag = bag.x_bags[bid]

			if sbag is None:
				sub_bags.append(None)
			else:
				res = self.sub_bags[bid](x_0, sbag) 
				sub_bags.append(res)

		return Result_Out(x, bag.x_len, sub_bags)

#==============================
class Net(torch.nn.Module):
	def __init__(self, meta):
		super().__init__()

		self.model_in  = BagInput(meta, LeakyReLU(), None)
		self.model_out = BagOutput(meta, root=True)

		self.model_cls_p = torch.nn.Linear(config.BAG_SIZE, config.CLASSES)		# class probs
		self.model_cls_a = torch.nn.Linear(config.BAG_SIZE, 1)					# classifying action
		self.model_v     = torch.nn.Linear(config.BAG_SIZE, 1)					# value function

		self.opt = torch.optim.AdamW(self.parameters(), lr=config.OPT_LR, weight_decay=config.OPT_L2)
		# self.opt = torch.optim.SGD(self.parameters(), lr=0.03)

		self.cls_loss = torch.nn.CrossEntropyLoss(reduction='none')

		self.meta = meta 

	def forward(self, batch, skip_action=False):
		res = self.model_in(batch, None)

		# print("X:", res.x)

		# outputs
		o_v   = self.model_v(res.x)
		o_cls_p = self.model_cls_p(res.x)

		if skip_action:
			return o_v, o_cls_p

		else:
			o_cls_a = self.model_cls_a(res.x)

			o_f	  = self.model_out(res.x, res)
			o_f.x = torch.cat( (o_cls_a, o_f.x), dim=1 )	# append o_cls to features

			# create per-sample result
			f = make_struct(o_f)

			return o_v, o_cls_p, f

	def get_loss(self, s, a, r, s_, y, net_res, flag):
		p = torch.ones(len(a), config.DATA_DEPTH)
		cls_p = []
		cls_y = []

		for i, e in enumerate(a):
			for l, e_ in enumerate(e):
				p[i, l] = e_[1]

				if l == 0 and e_[0][1] == 0: # it's a classification
					cls_p.append(e[l+1][1])
					cls_y.append(y[i])
					break

		pi = p.prod(dim=1)
		log_pi = torch.log(pi)

		r = torch.FloatTensor(r)
		flag = torch.FloatTensor(flag)

		# o_v = self(s, only_v=True).view(-1)
		o_v, full_cls_p, full_p = net_res
		o_v = o_v.view(-1)
		o_v_, _ = self(s_, skip_action=True)
		o_v_ = o_v_.view(-1)

		# print(r)
		# print(flag)
		# print(o_v_)
		# print(o_v)
		# exit()

		q = r + flag * config.GAMMA * o_v_.detach()
		# TODO - there was a gradient leak bug, check!

		adv = q - o_v
		v_err = q.clamp(max=1.0) - o_v

		# print(f'{p=}')
		# print(f'{pi=}')
		# print(f'{log_pi=}')

		# print(f'{r=}')
		# print(f'{adv=} {q=} {o_v=} {o_v_=}')
		# input()

		loss_pi = -adv.detach() * log_pi
		loss_v  = v_err ** 2
		# loss_h  = -entropy(full_p, self.meta) # full entropy
		loss_h  = log_pi * log_pi.detach() # sampled entropy
		entropy = -torch.mean(log_pi)

		if config.SAMPLE_CLS:
			# sample class loss when terminal action is selected
			if len(cls_y) > 0:
				cls_y = torch.LongTensor(cls_y)
				cls_p = torch.stack(cls_p)
				loss_cls = self.cls_loss(cls_p, cls_y)
			else:
				loss_cls = 0

		else:
			# weight class loss
			cls_a_p = torch.softmax( torch.stack([x.p.flatten() for x in full_p]), dim=1 ) [:, 0]
			loss_cls_w = torch.clamp(cls_a_p.detach() * config.ALPHA_CLS, 0., 1.)

			cls_y = torch.LongTensor(y)
			loss_cls = loss_cls_w * self.cls_loss(full_cls_p, cls_y)

		loss_pi  = torch.mean(loss_pi)
		loss_v   = torch.mean(loss_v)
		loss_h   = torch.mean(loss_h)
		loss_cls = torch.mean(loss_cls)

		return loss_pi, loss_v, loss_h, loss_cls, entropy

	def train_(self, s, a, r, s_, y, net_res, flag, info):
		loss_pi, loss_v, loss_h, loss_cls, entropy = self.get_loss(s, a, r, s_, y, net_res, flag)
		loss = config.W_PI * loss_pi + config.W_V * loss_v + config.W_H * loss_h + config.W_CLS * loss_cls

		# loss = torch.mean(loss_pi)
		# loss = torch.mean(loss_v)

		# debug
		# l_p = torch.mean(loss_pi).item() * config.W_PI
		# l_v = torch.mean(loss_v).item()  * config.W_V
		# l_h = torch.mean(loss_h).item()  * config.W_H
		# l   = loss.item()

		# print(l_p, l_v, l_h, l)
		# print(f"(Loss) PI: {loss_pi.item():.4f}, V: {loss_v.item():.4f}, H: {loss_h.item():.4f}\r", end="")

		self.opt.zero_grad()
		loss.backward()

		# clip the gradient norm
		grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), config.OPT_MAX_NORM)
		# if grad_norm > 0.1:
		# 	print(grad_norm)

		self.opt.step()

		return loss, loss_pi, loss_v, loss_h, loss_cls, entropy, grad_norm

	# use before pretraining
	def set_no_mask(self, flag_no_mask):
		self.model_in.set_no_mask(flag_no_mask)

	def pretrain_(self, s, y):
		v, y_ = self(s, skip_action=True)

		loss_cls = self.cls_loss(y_, y).mean()

		# lower bound for v
		y_pred = y_.argmax(dim=1)
		y_correct = (y_pred == y).to(torch.float32).detach().view(-1, 1)
		v_target = y_correct * config.REWARD_CORRECT + (1. - y_correct) * config.REWARD_INCORRECT
		loss_v = torch.mean( (v - v_target) ** 2 )

		# loss = config.W_CLS * loss_cls + config.W_V * loss_v # NOTE: disabled v pretraining
		loss = config.W_CLS * loss_cls

		self.opt.zero_grad()
		loss.backward()
		self.opt.step()

		return loss_cls.item(), loss_v.item()

	def save(self, file='model'):
		torch.save(self.state_dict(), file)

	def load(self, file='model'):	    
		self.load_state_dict(torch.load(file, map_location=config.DEVICE))

	def set_lr(self, lr):
		self.lr = lr

		for param_group in self.opt.param_groups:
			param_group['lr'] = lr
