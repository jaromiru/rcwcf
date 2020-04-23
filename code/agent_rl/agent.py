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
		v, cls_p, res = self.net(s)

		actions = []
		for i, it in enumerate(res):
			ag_act = []
			bag_meta = self.meta

			for l in range(config.DATA_DEPTH):
				p = torch.softmax(it.p.flatten(), dim=0)

				if self.greedy:
					a = torch.argmax(p, 0, keepdim=True).item()
				else:
					a = torch.multinomial(p, 1).item()

				p_a = p[a]
				a_i, a_f = to_matrix(a, it.p.shape[1])	# item, feature
				ag_act.append(( (a_i, a_f), p_a ))

				if l == 0:
					if a_f == 0:	# classification
						y_ = torch.argmax(cls_p[i]).item()
						ag_act.append((y_, cls_p[i]))
						break
					else:
						a_f -= 1

				bid = bag_meta.bag_idx[a_f]
				if bid is None:	# it is a feature
					break

				else:			# it is a bag
					it = it.bags[a_i][bid]
					bag_meta = bag_meta.bags[bid][1]

					if not it or len(it.p) == 0:	# not-expanded
						break

			actions.append(ag_act)

		return actions, (v, res)

	def step(self):
		s = self.s

		a, net_res = self.act(s)
		s_, r, y, flag, info = self.env.step(a)

		self.s = s_

		# the state s is overwritten in-place!
		return (None, a, r, s_, y, net_res, flag, info)
