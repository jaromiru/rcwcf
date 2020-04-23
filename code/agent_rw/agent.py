import numpy as np
import torch

from config import *

#==============================
def to_matrix(i, mlen):
	return i // mlen, i % mlen

#==============================
class Agent():
	def __init__(self, env, net, meta, greedy=False):
		self.env = env
		self.net = net
		self.meta = meta

		self.greedy = greedy

		self.s = env.reset()

	''' choose actions for each agent '''
	def act(self, s):
		actions = []
		for i, it in enumerate(s):
			ag_act = []
			bag_meta = self.meta

			# print(s[i])
			for l in range(config.DATA_DEPTH):
				if not isinstance(it, list):
					it = [it]

				m = torch.stack([x.mask for x in it])
				p = torch.ones(len(it), len(bag_meta.feat_idx))
				p[m == 1.0] = -np.inf
				p = torch.softmax(p.flatten(), dim=0)
				a = torch.multinomial(p, 1).item()

				a_i, a_f = to_matrix(a, len(bag_meta.feat_idx))	# item, feature
				ag_act.append((a_i, a_f))

				bid = bag_meta.bag_idx[a_f]
				if bid is None:	# it is a feature
					break

				else:			# it is a bag
					it = it[a_i].bags[bid]
					bag_meta = bag_meta.bags[bid][1]

					if not it:	# not-expanded
						break

			actions.append(ag_act)

		return actions

	def step(self):
		s = self.s

		a = self.act(s)
		s_, r, y, flag, info = self.env.step(a)

		self.s = s_

		return (s, a, r, s_, y, flag, info)
