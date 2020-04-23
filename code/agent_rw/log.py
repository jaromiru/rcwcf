import numpy as np

from env import ItemMask
from env import SeqEnv
from net import Net
from agent import Agent

from config import config

import time, torch
from itertools import compress

#==============================
class Log:
	def __init__(self, data, net, meta):
		self.data = data
		self.net  = net
		self.meta = meta

		self.watch = ItemMask(data[0], meta)

	def eval(self):
		env   = SeqEnv(self.data, self.meta)
		agent = Agent(env, self.net, self.meta)

		_fc = 0.
		_corr = 0.
		_tot = 0

		while True:
			s, a, r, s_, y, flag, info = agent.step()
			raw_cost = info[0]

			if np.all(flag == -1):
				break

			finished   = flag == 0.		# episode finished
			terminated = flag == -1 	# no more data

			_fc   += np.sum(raw_cost[~terminated])
			_tot  += np.sum(finished)
	
			if np.sum(finished) > 0:		
				s_finished = list(compress(s, finished))
				y_finished = torch.tensor(y[finished])

				y_ = torch.argmax(self.net(s_finished), dim=1)

				_corr += torch.sum(y_ == y_finished).item()

		_fc /= _tot
		_corr /= _tot

		# print("r: {:.3f}, acc: {:.3f}".format(_r, _corr))
		return _fc, _corr

	def info(self):
		v, cls_p, f = self.net([self.watch])

		v = v.item()
		f = f[0]

		def _to_p(f):
			f.p = f.p.flatten().softmax(0).numpy()

			for b in f.bags:
				for e in b:
					if e: _to_p(e)

		_to_p(f)

		print(f"V: {v:.4f}, CLS_P: {cls_p}, F: {f}")
