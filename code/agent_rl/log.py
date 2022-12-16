import numpy as np
import sklearn.metrics as metrics

from env import ItemMask
from env import SeqEnv
from net import Net
from agent import Agent

from config import config

import time, torch

#==============================
class Log:
	def __init__(self, name, data, net, meta):
		self.name = name
		self.data = data
		self.net  = net
		self.meta = meta

		self.watch = ItemMask(data[0], meta)

	def eval(self, greedy):
		env   = SeqEnv(self.data, self.meta)
		agent = Agent(env, self.net, self.meta, greedy)

		_r = 0.
		_fc = 0.
		_corr = 0
		_tot = 0

		true_y = []
		pred_y = []

		fc_log = []

		while True:
			s, a, r, s_, y, full_p, flag, info = agent.step()
			raw_cost = info[0]
			tot_cost = info[2]

			if np.all(flag == -1):
				break

			finished   = flag == 0.		# episode finished
			terminated = flag == -1 	# no more data

			_r 	  += np.sum(r[~terminated])
			_fc   += np.sum(raw_cost[~finished])
			_corr += np.sum(r[finished] == config.REWARD_CORRECT)
			_tot  += np.sum(finished)
			fc_log.extend(tot_cost[finished])

			# print(f"Evaluated ({self.name}): {_tot}/{len(self.data)}", end='\r')				

			for idx in np.argwhere(finished).flatten():
				true_y.append(y[idx])
				pred_y.append(info[1][idx])

			# from itertools import compress
			# cls_p = [x[1][1].softmax(0) for x in compress(a, finished)]
			# print("CLS_P:", cls_p)
			# print("Classified:", info[1][finished])

		_r /= _tot
		_fc /= _tot
		_corr /= _tot

		fc_hist = np.histogram(fc_log, bins=np.arange(np.max(fc_log)+2))

		print(f"Metrics ({self.name}):")
		print("R2:", metrics.r2_score(true_y, pred_y))
		print(metrics.classification_report(true_y, pred_y))
		print()
		print(f"r: {_r:.3f}, fc: {_fc:.3f}, acc: {_corr:.3f}")
		print()
		print(fc_hist)

		return _r, _fc, _corr

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
		cls_p = cls_p.flatten().softmax(0).numpy()
		print(f"V: {v:.4f}, CLS_P: {cls_p}, F: {f}")

		return v, cls_p, f
