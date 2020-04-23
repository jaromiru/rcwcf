import json, torch, random, argparse, sys
import numpy as np

from env import Env, SeqEnv
from net import Net
from agent import Agent
from log import Log
import utils, datalib

from config import config

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help="dataset name")
parser.add_argument('target', type=float, help="target lambda")

parser.add_argument('-device', type=str, choices=['auto', 'cpu', 'cuda'], default='cpu', help="Which device to use")
parser.add_argument('-seed', type=int, default=None, help="random seed")
parser.add_argument('-dataseed', type=int, default=1234, help="seed to shuffle data")

parser.add_argument('-model', type=str, default='run.mdl', help="load model from this file")

args = parser.parse_args()
args.batch = 1
args.epochs = 0
args.eplen = 0
args.log = 0

config.init(args)

#--- init seed
if config.SEED:
	np.random.seed(config.SEED)
	random.seed(config.SEED)
	torch.manual_seed(config.SEED)
	torch.cuda.manual_seed(config.SEED)

np.set_printoptions(threshold=np.inf, precision=3, suppress=True, linewidth=np.inf)

# --- load data
data, meta, meta_full = datalib.load_data(config.DATA_FILE, config.META_FILE)
config.init_dataset(meta_full)

print("config =", config)
print(f"Using dataset {meta_full['name']} with {meta_full['samples']} samples and {meta_full['classes']} classes.")

data_trn, data_val, data_tst, shuffle_idx = datalib.split(data, config.DATASEED)

net = Net(meta)
env = SeqEnv(data_trn, meta)
agent = Agent(env, net, meta)

net.load(config.MODEL_FILE)

# net.eval()
with torch.no_grad():
	while True:
		print("--------------")
		print(f"Data idx: {env.idx}, orig: {shuffle_idx[env.idx]}")
		print(f"Current sample: {env.x}")
		print("--------------")

		print("s: ", agent.s)
		print()

		o_v, o_cls_p, f = net(agent.s)
		print("CLS_P:", o_cls_p[0].softmax(0).numpy())
		print("ACT_P:", f[0].p[0].softmax(0).numpy())
		print("s_value: ", o_v[0])
		print()

		s, a, r, s_, y, net_res, flag, info = agent.step()

		print(f"a: {a}, r: {r}, cont: {flag}")
		print()

		print("s_: ", s_)

		if flag[0] == 0.:
			print(f"Classified: {info[1][0]} ({'correct' if y == info[1][0] else 'incorrect'})")

			print("=============================")

		input()

# tot = 0
# cor = 0

# while(True):
# 	s, a, r, s_, y, net_res, flag, info = agent.step()
# 	if flag == 0:
# 		tot += 1
		
# 		if r[0] == config.REWARD_CORRECT:
# 			cor += 1
	

# 		# y_ = a[0][1][0]
# 		# y = y[0]

# 		# if y == y_:
# 		# 	cor += 1

# 		print(f"\r{cor}/{tot}={cor/tot:.2f}")

