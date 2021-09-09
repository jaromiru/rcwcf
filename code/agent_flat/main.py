import numpy as np
import json, torch, random, argparse, sys, gc
import sklearn.metrics as metrics
import wandb, os

from env import RandomItemMask
from env import Env
from net import Net
from agent import Agent
from log import Log

import utils, datalib

from config import config

# NOTES
#
# * there has to be at least one feature revealed in a bag initially (due to mask=0)
#
#============================== main
np.set_printoptions(threshold=np.inf, precision=3, suppress=True, linewidth=np.inf)

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help="dataset name")
parser.add_argument('target', type=float, help="target lambda")

parser.add_argument('-device', type=str, choices=['auto', 'cpu', 'cuda'], default='cpu', help="Which device to use")
parser.add_argument('-ncpus', type=int, default=2, help="How many CPUs to use")
parser.add_argument('-seed', type=int, default=None, help="random seed")
parser.add_argument('-dataseed', type=int, default=1234, help="seed to shuffle data")

parser.add_argument('-epochs', type=int, default=100000, help="max train epochs")
parser.add_argument('-eplen', type=int, default=1000, help="steps per epoch")

parser.add_argument('-log', type=str, help="log raw results to a file")
parser.add_argument('-model', type=str, help="save model to this file")

parser.add_argument('-nopretrain', action='store_const', const=True, help="skip pretraining")
parser.add_argument('-load', action='store_const', const=True, help="load progress")
parser.add_argument('-fasteval', action='store_const', const=True, help="Skip training set")

parser.add_argument('-lr', type=float, default=1.0e-3, help="initial learning rate")
parser.add_argument('-alpha_h', type=float, default=0.05, help="initial entropy weight")
parser.add_argument('-alpha_h_min', type=float, default=0.0025, help="initial entropy weight")
parser.add_argument('-batch', type=int, default=256, help="batch size")

parser.add_argument('-alpha_cls', type=float, default=1.0, help="class weighting")
parser.add_argument('-sample_cls', action='store_const', const=True, help="sample classification loss")
parser.add_argument('-l2', type=float, default=1.0e-4, help="weight decay")

cmd_args = parser.parse_args()

config.init(cmd_args)

#--- init seed
if config.SEED:
	np.random.seed(config.SEED)
	random.seed(config.SEED)
	torch.manual_seed(config.SEED)
	torch.cuda.manual_seed(config.SEED)

# set ncpus
torch.set_num_threads(cmd_args.ncpus)	

wandb.init(project="rcwcf-flat", name=f'{cmd_args.dataset} {cmd_args.target}', config=config)
wandb.save("*.pt")
wandb.save("*.log")

if config.LOG_FILE is None: 
	config.LOG_FILE = os.path.join(wandb.run.dir, "run.log")

if config.MODEL_FILE is None:
	config.MODEL_FILE = os.path.join(wandb.run.dir, "model.pt")

# --- load data
data, meta, meta_full = datalib.load_data(config.DATA_FILE, config.META_FILE)
config.init_dataset(meta_full)

print("config =", config)
print(f"Using dataset {meta_full['name']} with {meta_full['samples']} samples and {meta_full['classes']} classes.")

data_trn, data_val, data_tst, _ = datalib.split(data, config.DATASEED)

net = Net(meta).to(config.DEVICE)
env = Env(data_trn, meta)
agent = Agent(env, net, meta)

log_trn = Log("train", data_trn, net, meta)
log_val = Log("validation", data_val, net, meta)
log_tst = Log("test", data_tst, net, meta)

print("Lambda:", config.LAMBDA)
print(net)

def decay_time(step, start, min, factor, rate):
	exp = step / rate * factor
	value = (start - min) / (1 + exp) + min

	return value

def decay_exp(step, start, min, factor, rate):
	exp = step / rate
	value = (start - min) * (factor ** exp) + min

	return value

def set_lr(ep_steps):
	lr = decay_exp(ep_steps, config.OPT_LR, config.OPT_LR_MIN, config.OPT_LR_FACTOR, config.LR_SCHEDULE)
	# ep = ep_steps // config.LR_SCHEDULE
	# lr = config.OPT_LR * (config.OPT_LR_FACTOR ** ep)
	net.set_lr(lr)

	print(f"LR: {lr:.2e}")

original_W_H = config.W_H
def set_h(ep_steps):
	# ep = ep_steps // config.H_SCHEDULE
	# config.W_H = original_W_H * (config.OPT_H_FACTOR ** ep)

	config.W_H = decay_time(ep_steps, original_W_H, config.H_MIN, config.OPT_H_FACTOR, config.H_SCHEDULE)
	print(f"W_H: {config.W_H}")

#---------------------------------------------------------------------
if cmd_args.load:
	print("Loading progress!")
	net.load(config.MODEL_FILE)
	ep_start = (sum(1 for line in open(config.LOG_FILE, 'r')) - 1) * config.EPOCH_STEPS
	set_lr(ep_start)
	set_h(ep_start)

else:
	ep_start = 1

if config.LOG_FILE:
	if cmd_args.load:		
		log_file = open(config.LOG_FILE, 'a')
	else:
		log_file = open(config.LOG_FILE, 'w')

#---------------------------------------------------------------------
if not cmd_args.nopretrain and not cmd_args.load:
	print("\nPre-training...")

	val_x = data_val
	val_y_np = np.array([x.label for x in val_x])
	val_y = torch.from_numpy(val_y_np).to(config.DEVICE)
	mov_avg = None

	for ep in range(100):
		for it in range(config.EPOCH_STEPS // 10):
			rand_p = np.random.rand()

			trn_x = random.choices(data_trn, k=config.AGENTS)
			trn_x = [RandomItemMask(x, meta, rand_p) for x in trn_x]
			# print(meta.feat_labels)
			# print(trn_x[0])
			# input()

			trn_y = torch.tensor([x.label for x in trn_x]).to(config.DEVICE)
			loss_cls, loss_v = net.pretrain_(trn_x, trn_y)
			sys.stdout.write('.')
			sys.stdout.flush()

		# evaluate & check early-stop
		net.set_no_mask(True)
		net.eval()
		val_y_ = net(val_x, skip_action=True)[1]
		val_y_np_ = val_y_.argmax(dim=1).detach().numpy()
		net.train()
		net.set_no_mask(False)

		loss_cls_val = net.cls_loss(val_y_, val_y).mean().item()
		
		if mov_avg is None:
			mov_avg = loss_cls_val
		else:
			mov_avg = 0.95 * mov_avg + 0.05 * loss_cls_val

		# print(f"TRN_LOSS: {loss:.4f} | VAL_LOSS: {loss_cls_val:.4f}")
		print()
		print(f"{loss_cls=:.4f} {loss_v=:.4f} {loss_cls_val=:.4f} {mov_avg=:.4f}")
		print(f"Validation metrics:")
		print(metrics.classification_report(val_y_np, val_y_np_))
		print()

		if ep >= 10 and mov_avg <= loss_cls_val:
			print("Terminating pre-training")
			break


#---------------------------------------------------------------------
print("\nTraining...")

fps = utils.Fps()
fps.start()
set_lr(ep_start)

for ep_steps in range(ep_start, config.TRAINING_EPOCHS+1):
	*trj, = agent.step()
	loss, loss_pi, loss_v, loss_h, loss_cls, entropy, grad_norm = net.train_(*trj)

	sys.stdout.write('.')
	sys.stdout.flush()

	if utils.is_time(ep_steps, config.LR_SCHEDULE):
		set_lr(ep_steps)

	if utils.is_time(ep_steps, config.H_SCHEDULE):
		set_h(ep_steps)

	if utils.is_time(ep_steps, config.EPOCH_STEPS):
		with torch.no_grad():
			print()

			net.save(config.MODEL_FILE)

			net.eval()	# eval-mode
			fps_ = fps.fps(ep_steps)

			if cmd_args.fasteval:
				trn_r, trn_fc, trn_acc = (0, 0, 0) 
			else:
				trn_r, trn_fc, trn_acc = log_trn.eval(greedy=False)

			tst_r, tst_fc, tst_acc = log_tst.eval(greedy=False)
			val_r, val_fc, val_acc = log_val.eval(greedy=False)

			print(f"ep_steps {ep_steps}: TRN: {trn_r:.2f}/{trn_fc:.2f}/{trn_acc:.2f} | VAL: {val_r:.2f}/{val_fc:.2f}/{val_acc:.2f} | TST: {tst_r:.2f}/{tst_fc:.2f}/{tst_acc:.2f}, FPS: {fps_:.2f}")
			v, cls_p, f = log_trn.info()

			if config.LOG_FILE:
				print(f"{trn_r:.4f} {trn_fc:.4f} {trn_acc:.4f} {val_r:.4f} {val_fc:.4f} {val_acc:.4f} {tst_r:.4f} {tst_fc:.4f} {tst_acc:.4f}", file=log_file, flush=True)

			net.train()	# train-mode

			# log wandb
			log = {
				'trn_r': trn_r,
				'trn_fc': trn_fc,
				'trn_acc': trn_acc,

				'val_r': val_r,
				'val_fc': val_fc,
				'val_acc': val_acc,

				'tst_r': tst_r,
				'tst_fc': tst_fc,
				'tst_acc': tst_acc,

				'rate': fps_,
				'loss': loss,
				'loss_pi': loss_pi,
				'loss_v': loss_v,
				'loss_h': loss_h,
				'loss_cls': loss_cls,
				'entropy': entropy,
				'grad_norm': grad_norm,
				
				'value': v,
				'cls_p': cls_p,
				'f_p': f.p,

				'lr': net.lr,
				'alpha_h': config.W_H
			}

			print(log)
			wandb.log(log)
