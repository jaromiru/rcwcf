import numpy as np
import json, random, torch, copy

from config import config

#==============================
class ItemMask:
	def __init__(self, item, meta):
		self.orig_item = item

		self.feats 		= torch.zeros_like(item.feats)
		self.mask		= meta.init_mask.clone()
		self.bags 		= []

		for f, m in enumerate(self.mask):
			# include the features
			if m > 0:
				self.feats[meta.feat_idx[f][0]:meta.feat_idx[f][1]] = item.feats[meta.feat_idx[f][0]:meta.feat_idx[f][1]]

			# process bags
			bid = meta.bag_idx[f]
			if bid is not None: # it is a BAG
				if m > 0 or meta.costs[f] == 0:
					bag_items = [ItemMask(it, meta.bags[bid][1]) for it in item.bags[bid]]
				else:
					bag_items = []

				self.bags.append(bag_items)

		if hasattr(item, 'label'):
			self.label = item.label

	def reveal(self, actions, meta):
		_, f = actions[0]

		bid = meta.bag_idx[f]
		if bid is None:
			assert self.mask[f] == 0.0

			self.mask[f] = 1.0
			self.feats[meta.feat_idx[f][0]:meta.feat_idx[f][1]] = self.orig_item.feats[meta.feat_idx[f][0]:meta.feat_idx[f][1]]

			return meta.costs[f]

		else: 	# it is a bag
			if not self.bags[bid]:	# expand
				assert len(actions) == 1
				assert self.mask[f] == 0.

				if self.orig_item.bags[bid]:
					self.bags[bid] = [ItemMask(it, meta.bags[bid][1]) for it in self.orig_item.bags[bid]]
					
					all_mask = torch.cat([x.mask for x in self.bags[bid]])
					self.mask[f] = all_mask.mean()
				else:
					self.bags[bid] = []
					self.mask[f] = 1.0

				return meta.costs[f]

			else:	# forward
				assert len(actions) >= 1
				assert self.mask[f] < 1.

				it, _ = actions[1]
				r = self.bags[bid][it].reveal(actions[1:], meta.bags[bid][1])

				all_mask = torch.cat([x.mask for x in self.bags[bid]])
				self.mask[f] = all_mask.mean()

				return r

	def __str__(self):
		return f"{{feats: {self.feats}, mask: {self.mask}, bags: {self.bags}}}\n"

	__repr__ = __str__

#==============================
class Env:
	def __init__(self, data, meta):
		self.data = data
		self.meta = meta

	def reset(self):
		self.x  = random.choices(self.data, k=config.AGENTS)
		self.x_ = [ItemMask(it, self.meta) for it in self.x]
		self.budget = np.zeros(config.AGENTS) 

		return self.x_.copy()	# correct would be copy.deepcopy()!

	def _generate_sample(self):
		raw_it = random.choice(self.data)
		msk_it = ItemMask(raw_it, self.meta)

		return raw_it, msk_it

	def step(self, actions):
		raw_cost = np.zeros(config.AGENTS, dtype=np.float32)	
		flag = np.ones(config.AGENTS, dtype=np.float32)		# non-termination flag
		old_y = np.array([x.label for x in self.x])

		for i in range(config.AGENTS):
			raw_cost[i] = self.x_[i].reveal(actions[i], self.meta)
			complete = self.x_[i].mask.mean() == 1.0

			self.budget[i] += raw_cost[i]

			if self.budget[i] >= config.BUDGET or complete:
				self.x[i], self.x_[i] = self._generate_sample()
				self.budget[i] = 0
				flag[i] = 0.0

		return (self.x_.copy(), 0, old_y, flag, (raw_cost, ))	# correct would be copy.deepcopy()!

# TODO
class SeqEnv(Env):
	def __init__(self, data, meta):
		super().__init__(data, meta)

	def reset(self):
		self.budget = np.zeros(config.AGENTS) 

		if len(self.data) < config.AGENTS:
			self.x = []
			self.x_ = []
			self.idx = 0

			for i in range(config.AGENTS):
				x, x_ = self._generate_sample()

				self.x.append(x)
				self.x_.append(x_)

		else:
			self.x = self.data[:config.AGENTS]	
			self.x_ = [ItemMask(it, self.meta) for it in self.x]
			self.idx = config.AGENTS

		return self.x_	# correct would be copy.deepcopy()!

	def _generate_sample(self):
		if self.idx >= len(self.data):
			x = self.data[0] 
			x_ = ItemMask(self.data[0], self.meta)
			x_.dummy = True

			return x, x_

		else:
			x = self.data[self.idx]
			self.idx += 1

			return x, ItemMask(x, self.meta)

	def step(self, actions):
		term = np.array( [hasattr(self.x_[i], "dummy") for i in range(config.AGENTS)])
		x_, r, y, flag, info = super(SeqEnv, self).step(actions)

		flag[term] = -1
		info[0][term] = 0

		return x_, r, y, flag, info