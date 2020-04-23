import numpy as np
import json, torch, random, argparse, sys

from env import Env
from net import Net
from agent import Agent
from log import Log

import utils, datalib

from config import config
from itertools import compress
# NOTES
#
# * there has to be at least one feature revealed in a bag initially (due to mask=0)
#
#============================== main
np.set_printoptions(threshold=np.inf, precision=3, suppress=True, linewidth=np.inf)

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help="dataset name")
parser.add_argument('budget', type=float, help="target budget")

parser.add_argument('-device', type=str, choices=['auto', 'cpu', 'cuda'], default='cpu', help="Which device to use")
parser.add_argument('-seed', type=int, default=None, help="random seed")
parser.add_argument('-dataseed', type=int, default=1234, help="seed to shuffle data")

parser.add_argument('-batch', type=int, default=128, help="batch size")
parser.add_argument('-epochs', type=int, default=100000, help="max train epochs")
parser.add_argument('-eplen', type=int, default=1000, help="steps per epoch")

parser.add_argument('-log', type=str, default='run.log', help="log raw results to a file")
parser.add_argument('-model', type=str, default='run.mdl', help="save model to this file")

parser.add_argument('-load', action='store_const', const=True, help="load progress")

args = parser.parse_args()

config.init(args)

#--- init seed
if config.SEED:
	np.random.seed(config.SEED)
	random.seed(config.SEED)
	torch.manual_seed(config.SEED)
	torch.cuda.manual_seed(config.SEED)

# --- load data
data, meta, meta_full = datalib.load_data(config.DATA_FILE, config.META_FILE)
config.init_dataset(meta_full)

print("config =", config)
print(f"Using dataset {meta_full['name']} with {meta_full['samples']} samples and {meta_full['classes']} classes.")

data_trn, data_val, data_tst = datalib.split(data, config.DATASEED)

net = Net(meta).to(config.DEVICE)
env = Env(data_trn, meta)
agent = Agent(env, net, meta)

log_trn = Log(data_trn, net, meta)
log_val = Log(data_val, net, meta)
log_tst = Log(data_tst, net, meta)

print(net)

fps = utils.Fps()
fps.start()

def set_lr(ep_steps):
	ep = ep_steps // (config.EPOCH_STEPS * 10)
	lr = config.OPT_LR * (config.OPT_LR_FACTOR ** ep)
	
	net.set_lr(lr)
	print(f"LR: {lr:.2e}")

if config.LOG_FILE:
	if args.load:		
		log_file = open(config.LOG_FILE, 'a')
	else:
		log_file = open(config.LOG_FILE, 'w')

flatten = lambda l: [item for sublist in l for item in sublist]

finished_s = []
finished_y = []
finished_cnt = 0

print("\nTraining...")
for ep_steps in range(1, config.TRAINING_EPOCHS):
	if utils.is_time(ep_steps, config.EPOCH_STEPS):
		with torch.no_grad():
			fps_ = fps.fps(ep_steps)

			# trn_r, trn_fc, trn_acc = (0, 0, 0) 
			trn_fc, trn_acc = log_trn.eval()
			val_fc, val_acc = log_val.eval()
			tst_fc, tst_acc = log_tst.eval()
			# tstg_r, tstg_fc, tstg_acc = (0, 0, 0)

			print()
			print(f"ep_steps {ep_steps}: TRN: {trn_fc:.2f}/{trn_acc:.2f} | VAL: {val_fc:.2f}/{val_acc:.2f} | TST: {tst_fc:.2f}/{tst_acc:.2f} | FPS: {fps_:.2f}")

			if config.LOG_FILE:
				print(f"{trn_acc:.2f} {trn_fc:.2f} {trn_acc:.2f} {val_acc:.2f} {val_fc:.2f} {val_acc:.2f} {tst_acc:.2f} {tst_fc:.2f} {tst_acc:.2f}", file=log_file, flush=True)

			net.save(config.MODEL_FILE)

	if utils.is_time(ep_steps, config.EPOCH_STEPS * 10):
		set_lr(ep_steps)

	# if utils.is_time(epoch, config.EPOCH_STEPS):
	# 	with torch.no_grad():
	# 		fps_ = fps.fps(epoch)

	# 		trn_acc = eval(data_trn, data_trn_y)
	# 		val_acc = eval(data_val, data_val_y)
	# 		tst_acc = eval(data_tst, data_tst_y)

	# 		# check for improvement
	# 		if trn_acc > best_acc:
	# 			fails = 0
	# 			best_acc = trn_acc
	# 			impr = "<"
	# 		else:		
	# 			impr = "."

	# 			fails += 1
	# 			if fails >= 5:
	# 				fails = 0
	# 				lr *= config.OPT_LR_FACTOR

	# 				print("\nNew LR:", lr)
	# 				net.set_lr(lr)

	# 				impr = "o"

	# 		print()
	# 		print(f"Epoch {epoch}: acc (trn/val/tst): {trn_acc:.4f}/{val_acc:.4f}/{tst_acc:.4f}, FPS: {fps_:.2f} {impr}")
	# 		print(f"{trn_acc:.4f} {val_acc:.4f} {tst_acc:.4f}", file=log_file, flush=True)

	while finished_cnt < config.AGENTS:
		s, a, r, s_, y, flag, info = agent.step()
		finished = flag == 0.		# episode finished

		finished_s.append( compress(s, finished) )
		finished_y.append( y[finished] )
		finished_cnt += np.sum(finished)

	net.train(flatten(finished_s), flatten(finished_y))

	finished_cnt = 0
	finished_s = []
	finished_y = []

	sys.stdout.write('.')
	sys.stdout.flush()

		# TODO finish the evaluation!


