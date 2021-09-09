from config import config
import json, numpy as np
import sys, collections, torch, itertools, os, pickle, random

class Item:
	def __init__(self, feats, bags, label):
		self.feats 		= feats
		self.bags 		= bags

		if label is not None:
			self.label = label

	def __str__(self):
		return f"{{feats: {self.feats}, bags: {self.bags}}}\n"

	__repr__ = __str__

class MetaItem:
	def __init__(self, feat_idx, feat_labels, bags, bag_idx, costs, init_mask):
		self.feat_idx 	 = feat_idx
		self.feat_labels = feat_labels
		self.bags 		 = bags
		self.bag_idx	 = bag_idx
		self.costs		 = costs

		self.init_mask   = init_mask

	def __str__(self):
		return f"{{feat_labels: {self.feat_labels}, feat_idx: {self.feat_idx}, mask: {self.init_mask}, costs: {self.costs}, bags: {self.bags}, bag_idx: {self.bag_idx}}}\n"

	__repr__ = __str__

def to_categorical(prop, values):
	index = values.index(prop)

	x = np.zeros(len(values))
	x[values.index(prop)] = 1

	return x

def to_bin13(prop):
	return list(map(float, np.binary_repr(prop, 13)))

def ip2bin(prop):
	try:
		bt = [int(x) for x in prop.split('.')]
		# v = bt[0] << 24 | bt[1] << 16 | bt[2] << 8 | bt[3]
		v = bt[0] ^ bt[1] ^ bt[2] ^ bt[3]
		return list(map(float, np.binary_repr(v, 8)))

	except:
		return [0.] * 8

# TODO: for string of length < 3 it returns all zeros
HIST_SIZE = 13
def to_trigram_hist(string):
	# print(string, print(string.encode())
	hist = [0] * HIST_SIZE

	string = string.encode()
	for s in range(0, len(string) - 2):
		v = (string[s] * 65536 + string[s+1] * 256 + string[s+2]) % HIST_SIZE
		hist[v] += 1

	hist = np.array(hist)
	hist_mag = np.sum(hist)

	if hist_mag > 0:
		hist = hist / hist_mag 	# normalize

	return hist

def process_bag(bag, meta):
	bag_items = []

	for item in bag:
		feats = []
		bags = []
		label = None

		for prop in meta:
			if prop in ['type', 'cost']:
				continue

			m_prop = meta[prop]
			i_prop = item[prop]
			
			if m_prop['type'] in ['binary', 'float']:
				feats.append(i_prop)

			elif m_prop['type'] in ['float_array_13', 'float_array_31']:
				if m_prop['type'] == 'float_array_13': p_size = 13
				if m_prop['type'] == 'float_array_31': p_size = 31

				assert len(i_prop) == p_size

				feats.extend(i_prop)

			elif m_prop['type'] == "str2trigram":
				feats.extend( to_trigram_hist(i_prop) )

			elif m_prop['type'] == "bin13":
				feats.extend( to_bin13(i_prop) )

			elif m_prop['type'] == "ip2bin":
				feats.extend( ip2bin(i_prop) )

			elif m_prop['type'] == "category":
				catv = to_categorical(i_prop, m_prop['values'])
				feats.extend(catv)

			elif m_prop['type'] == "bag":
				bagv = process_bag(i_prop, m_prop)
				feats.extend([0] * config.BAG_SIZE)
				bags.append(bagv)

			# elif m_prop['type'] == "bag_raw":	# some refactoring welcome!	
			# 	if m_prop['bag_type'] == "bin13":
			# 		bagv = [Item(torch.FloatTensor(to_bin13(x)), [], None) for x in i_prop]
			# 	else:
			# 		raise("Not recognized bag_type:", m_prop['bag_type'])

			# 	feats.extend([0] * config.BAG_SIZE)
			# 	bags.append(bagv)

			elif m_prop['type'] == "label":
				label = i_prop

			elif m_prop['type'] == "label_svm":
				label = 0 if i_prop == -1 else 1

		feats = torch.FloatTensor(feats)
		bag_items.append( Item(feats, bags, label) )

	return bag_items

def process_meta(meta):
	feat_idx = []
	feat_labels = []
	costs = []
	init_mask = []	
	bags = []
	bag_idx = []
	idx = 0
	fid = 0
	bid = 0

	for prop, m_prop in meta.items():
		if prop in ['type', 'cost']:	# keywords
			continue

		if m_prop['type'] in ['label', 'label_svm', 'skip']:
			continue

		elif m_prop['type'] in ['binary', 'float']:
			feat_idx.append( (idx, idx+1) )
			feat_labels.append(prop)
			costs.append(m_prop['cost'])
			init_mask.append(m_prop['cost'] == 0)
			bag_idx.append(None)

			idx += 1
			fid += 1

		elif m_prop['type'] in ['float_array_13', 'float_array_31']:
			if m_prop['type'] == 'float_array_13': p_size = 13
			if m_prop['type'] == 'float_array_31': p_size = 31

			feat_idx.append( (idx, idx + p_size) )
			feat_labels.append(prop)
			costs.append(m_prop['cost'])
			init_mask.append(m_prop['cost'] == 0)
			bag_idx.append(None)

			idx += p_size
			fid += 1

		elif m_prop['type'] == "str2trigram":
			feat_idx.append( (idx, idx + HIST_SIZE) )
			feat_labels.append(prop)
			costs.append(m_prop['cost'])
			init_mask.append(m_prop['cost'] == 0)
			bag_idx.append(None)

			idx += HIST_SIZE
			fid += 1

		elif m_prop['type'] == "bin13":
			feat_idx.append( (idx, idx + 13) )
			feat_labels.append(prop)
			costs.append(m_prop['cost'])
			init_mask.append(m_prop['cost'] == 0)
			bag_idx.append(None)

			idx += 13
			fid += 1

		elif m_prop['type'] == "ip2bin":
			feat_idx.append( (idx, idx + 8) )
			feat_labels.append(prop)
			costs.append(m_prop['cost'])
			init_mask.append(m_prop['cost'] == 0)
			bag_idx.append(None)

			idx += 8
			fid += 1

		elif m_prop['type'] == "category":
			feat_idx.append( (idx, idx + len(m_prop['values'])) )
			feat_labels.append(prop)
			costs.append(m_prop['cost'])
			init_mask.append(m_prop['cost'] == 0)
			bag_idx.append(None)

			idx += len(m_prop['values'])
			fid += 1

		elif m_prop['type'] == "bag":
			bagv = process_meta(m_prop)
			feat_idx.append( (idx, idx + config.BAG_SIZE) )
			feat_labels.append(prop)
			bags.append((fid, bagv))
			costs.append(m_prop['cost'])
			init_mask.append(torch.mean(bagv.init_mask) if m_prop['cost'] == 0 else 0.)
			bag_idx.append(bid)

			idx += config.BAG_SIZE
			fid += 1
			bid += 1

		else:
			assert False, "Unknown property " + m_prop['type']

		# elif m_prop['type'] == "bag_raw":
		# 	if m_prop['bag_type'] == "bin13":
		# 		bagv = MetaItem([(0, 13)], [prop+'_raw'], [], [None], torch.tensor([0.]), torch.tensor([1.]))
		# 	else:
		# 		raise("Not recognized bag_type:", m_prop['bag_type'])

		# 	feat_idx.append( (idx, idx + config.BAG_SIZE) )
		# 	feat_labels.append(prop)
		# 	bags.append((fid, bagv))
		# 	costs.append(m_prop['cost'])

		# 	init_mask.append(1.)
		# 	bag_idx.append(bid)

		# 	idx += config.BAG_SIZE
		# 	fid += 1
		# 	bid += 1

	costs = torch.tensor(costs)
	init_mask = torch.FloatTensor(init_mask)

	return MetaItem(feat_idx, feat_labels, bags, bag_idx, costs, init_mask)

def load_data(data_file, meta_file):
	data = json.load( open(data_file, "r") )
	meta = json.load( open(meta_file, "r"), object_pairs_hook=collections.OrderedDict )

	data_processed = process_bag(data, meta['description'])
	meta_processed = process_meta(meta['description'])

	# print(meta_processed)
	# print(data_processed[0])
	# print(data[0])
	# exit()

	return data_processed, meta_processed, meta

def split(data, data_seed):
	shuffle_idx = np.arange(len(data))

	if data_seed != 0:
		random.shuffle(shuffle_idx, random.Random(data_seed).random)
		data = np.array(data)[shuffle_idx]

	data_tst = data[:config.TEST_SAMPLES]
	data_val = data[config.TEST_SAMPLES:config.TEST_SAMPLES+config.VAL_SAMPLES]
	data_trn = data[config.TEST_SAMPLES+config.VAL_SAMPLES:]

	return data_trn, data_val, data_tst, shuffle_idx


# def load_data(data_file, meta_file, force=False):
# 	os.makedirs('cache/', mode=0o776, exist_ok=True)
# 	cache_file_data = f'cache/{config.DATASET}_data'

# 	if (not force) and os.path.exists(cache_file_data) and (os.path.getmtime(cache_file_data) >= os.path.getmtime(data_file)):
# 		print("Using cached data.")

# 		with open(cache_file_data, 'rb') as file:
# 			data_processed = pickle.load(file)

# 		meta = json.load( open(meta_file, "r"), object_pairs_hook=collections.OrderedDict )
# 		meta_processed = process_meta(meta['description'])

# 	else:
# 		print("Loading a new dataset.")

# 		data = json.load( open(data_file, "r") )
# 		meta = json.load( open(meta_file, "r"), object_pairs_hook=collections.OrderedDict )

# 		data_processed = process_bag(data, meta['description'])
# 		meta_processed = process_meta(meta['description'])

# 		with open(cache_file_data, 'wb') as file:
# 			pickle.dump(data_processed, file)

# 	return data_processed, meta_processed, meta

# TEST
if __name__ == '__main__':
	config.BAG_SIZE = 64
	data, meta_p, meta = load_data('../data/web_small.json', '../data/web_small_meta.json')

