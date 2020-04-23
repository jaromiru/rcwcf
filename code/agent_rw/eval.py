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
parser.add_argument('budget', type=float, help="target budget")

parser.add_argument('-device', type=str, choices=['auto', 'cpu', 'cuda'], default='cpu', help="Which device to use")
parser.add_argument('-seed', type=int, default=None, help="random seed")
parser.add_argument('-dataseed', type=int, default=1234, help="seed to shuffle data")

parser.add_argument('-model', type=str, default='model', help="load model from this file")

args = parser.parse_args()
args.batch = 1
args.epochs = None
args.eplen = None
args.log = None

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

data_trn, data_val, data_tst = datalib.split(data, config.DATASEED)

net = Net(meta)
env = SeqEnv(data_trn, meta)
agent = Agent(env, net, meta)

net.load(config.MODEL_FILE)

# net.eval()
with torch.no_grad():
	while True:
		print("--------------")

		print("s: ", agent.s)
		print()

		print(agent.s)
		cls_s0 = net(agent.s).softmax(0).numpy()

		s, a, r, s_, y, flag, info = agent.step()
		
		print(s)
		print("s_: ", s_)
		print(f"a: {a}, r: {r}, cont: {flag}")

		# cls_s1 = net(s_).softmax(0).numpy()

		# print("CLS_P:", o_cls_p[0].softmax(0).numpy())
		# print()


		input()
