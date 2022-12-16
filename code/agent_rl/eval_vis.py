import json, torch, random, argparse, sys, copy
import numpy as np

from env import Env, SeqEnv
from net import Net
from agent import Agent
from log import Log
import utils, datalib

from config import config

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help="dataset name")
parser.add_argument('-target', default=0, type=float, help="target lambda")

parser.add_argument('-device', type=str, choices=['auto', 'cpu', 'cuda'], default='cpu', help="Which device to use")
parser.add_argument('-seed', type=int, default=None, help="random seed")
parser.add_argument('-dataseed', type=int, default=1234, help="seed to shuffle data")

parser.add_argument('-model', type=str, default='run.mdl', help="load model from this file")
parser.add_argument('-emb_size', type=int, default=128, help="batch size")

args = parser.parse_args()
args.batch = 1
args.epochs = 0
args.eplen = 0
args.log = 0
args.lr = 0
args.l2 = 0
args.sample_cls = None
args.alpha_cls = 0
args.alpha_h = 0
args.alpha_h_min = 0

config.init(args)

#--- init seed
if config.SEED:
	np.random.seed(config.SEED)
	random.seed(config.SEED)
	torch.manual_seed(config.SEED)
	torch.cuda.manual_seed(config.SEED)

np.set_printoptions(threshold=np.inf, precision=3, suppress=True, linewidth=np.inf)

def sample_to_dict(meta, x):
	return x

def a_to_str(actions, meta):
	print(actions[0])
	if actions[0][0][1] == 0:
		return ["terminate", actions[1][0], actions[1][1]]

	actions = [[x[0][0], x[0][1], x[1]] for x in actions]
	actions[0][1] -= 1

	a_str = []
	for a in actions:
		smp, f, p = a
		bid = meta.bag_idx[f]

		f_label = meta.feat_labels[f]

		a_str.append((smp, f_label, p.item()))

		if bid is not None:
			meta = meta.bags[bid][1]

	return a_str

# {p: tensor([0.0127, 0.0589, 0.0394, 0.1231, 0.0362, 0.0695, 0.4574, 0.0319, 0.0591, 0.1118]), 
# 	bags: [(
# 		None, 
# 		{p: tensor([[-0.1059,  0.0661,  0.0072, -0.0563, -0.0709,  0.0402,  0.0429,  0.0852],
# 			        [-0.1059,  0.0661,  0.0072, -0.0563, -0.0709,  0.0402,  0.0429,  0.0852],
# 			        [-0.1059,  0.0661,  0.0072, -0.0563, -0.0709,  0.0402,  0.0429,  0.0852],
# 			        [-0.1059,  0.0661,  0.0072, -0.0563, -0.0709,  0.0402,  0.0429,  0.0852]]),
# 	    bags: [(None, None), (None, None), (None, None), (None, None)]}
# 	)]
# }
def augment_mask_probs(sample, state, probs, meta):
	for prop, value in list(sample.items()):
			if prop not in meta.feat_labels:
				print("skipping", prop)
				del sample[prop]
				continue

			fid = meta.feat_labels.index(prop)
			bid = meta.bag_idx[fid]

			prob = probs.p[fid].item()
			mask = state.mask[fid].item()

			sample[prop] = {'value': value, 'prob': prob, 'mask': mask}

			if bid is not None:
				bag_prob  = probs.bags[bid]
				if bag_prob is None or len(bag_prob.p) == 0: 
					sample[prop]['value'] = None
					continue

				# compute softmax over all whole bag
				bag_p = torch.softmax(bag_prob.p.flatten(), dim=0).view_as(bag_prob.p)

				# for all items go deeper
				for idx, it in enumerate(value):
					it_probs = utils.Object()
					it_probs.p = bag_p[idx]
					it_probs.bags = bag_prob.bags[idx] if len(bag_prob.bags) > 0 else None

					meta_bag = meta.bags[bid][1]
					it_state = state.bags[bid][idx]

					augment_mask_probs(it, it_state, it_probs, meta_bag)


def augment_sample(sample, state, probs, actions):
	sample = copy.deepcopy(sample)

	all_p = probs.p[0].softmax(0)
	at_p = all_p[0]
	ft_p = all_p[1:]

	probs.p = ft_p
	probs.bags = probs.bags[0]

	del sample[label_id]

	augment_mask_probs(sample, state, probs, meta)
	sample['terminate'] = {'prob': at_p.item(), 'mask': 0, 'value': None}
	
	# indicate the selected action
	if actions[0][0][1] == 0: # terminating action
		predicted_class = actions[1][0]
		sample['terminate']['selected'] = True
		sample['terminate']['mask'] = 1.0
		sample['terminate']['value'] = f'class={predicted_class}'

	else:
		sample_path = [sample]
		meta_path = meta
		actions = [[x[0][0], x[0][1], x[1]] for x in actions]
		actions[0][1] -= 1

		for a in actions:
			smp, f, p = a
			f_label = meta_path.feat_labels[f]
			bid = meta_path.bag_idx[f]

			smpl = sample_path[smp][f_label]
			smpl['selected'] = True

			# pprint(a)
			# pprint(meta_path)
			# pprint(smpl)

			if bid is not None:
				sample_path = smpl['value']
				meta_path = meta_path.bags[bid][1]

	return sample

# --- load data
# data_raw = json.load( open(f"../data/stats_raw_full.json", "r") )
data_raw = json.load( open(f"../data/{args.dataset}.json", "r") )

data, meta, meta_full = datalib.load_data(config.DATA_FILE, config.META_FILE)
config.init_dataset(meta_full)

label_id = meta_full['label']
label_type = meta_full['description'][label_id]['type']

print("config =", config)
print(f"Using dataset {meta_full['name']} with {meta_full['samples']} samples and {meta_full['classes']} classes.")

data_trn, data_val, data_tst, shuffle_idx = datalib.split(data, config.DATASEED)

net = Net(meta)
env = SeqEnv(data_tst, meta)
agent = Agent(env, net, meta)

net.load(config.MODEL_FILE)

from pprint import pprint
# net.eval()
with torch.no_grad():
	step = 0
	tot_cst = 0
	all_samples = []
	step_dict = []

	while True:
		orig_id = shuffle_idx[env.idx-1]
		# orig_id = shuffle_idx[config.TEST_SAMPLES+config.VAL_SAMPLES+env.idx-1] # for train set
		# print("--------------")
		print(f"Data idx: {env.idx}, orig: {orig_id}, step: {step}")

		sample = data_raw[orig_id]
		o_v, o_cls_p, f = net(agent.s)

		orig_s = copy.deepcopy(agent.s)
		orig_y = env.x[0].label
		s, a, r, s_, y, net_res, flag, info = agent.step()

		next_s = copy.deepcopy(agent.s)
		aug_s = orig_s if not flag else next_s

		cst = np.asscalar(info[0][0])

		if label_type == "label_svm":
			true_class = 0 if sample[label_id] == -1 else 1
		else:
			true_class = sample[label_id]

		sample_aug = augment_sample(sample, orig_s[0], f[0], a[0])
		cls_probs = o_cls_p[0].softmax(0).numpy().tolist()
		s_value = o_v[0].item()

		step_dict.append({'sample': sample_aug, 'cls_probs': cls_probs, 's_value': s_value, 'orig_id': int(orig_id), 'true_y': true_class, 'last_cost': cst, 'total_cost': tot_cst})
		tot_cst += cst

		# pprint(sample_aug)

		if flag[0] == 0:
			step = 0
			tot_cst = 0
			all_samples.append(step_dict)
			step_dict = []

			if len(all_samples) >= 100:
				break

		elif flag[0] == -1:
			break

		else:
			step += 1

	with open(f'debug/data_{args.dataset}.js', 'w') as file:
		step_obj = json.dumps(all_samples)

		print(f"dataset_name='{args.dataset}'", file=file)
		print(f"class_labels={meta_full['class_labels']}", file=file)
		print(f"data={step_obj}", file=file)
