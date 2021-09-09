import numpy as np
import json, torch, random, argparse, sys, wandb
import sklearn.metrics as metrics

from net import Net
import utils, datalib

from config import config

#============================== main
np.set_printoptions(threshold=np.inf, precision=3, suppress=True, linewidth=np.inf)

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help="dataset name")
parser.add_argument('-device', type=str, choices=['auto', 'cpu', 'cuda'], default='cpu', help="Which device to use")
parser.add_argument('-seed', type=int, default=None, help="random seed")
parser.add_argument('-dataseed', type=int, default=1234, help="seed to shuffle data")
parser.add_argument('-epochs', type=int, default=1000, help="max train epochs")
parser.add_argument('-eplen', type=int, default=100, help="steps per epoch")

parser.add_argument('-batch', type=int, default=128, help="batch size")
parser.add_argument('-log', type=str, default="run.log", help="log raw results to a file")
parser.add_argument('-lr', type=float, default=3.0e-3, help="initial learning rate")
parser.add_argument('-l2', type=float, default=1.0e-1, help="weight decay")

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

print(f"Using dataset {meta_full['name']} with {meta_full['samples']} samples and {meta_full['classes']} classes.")

data_trn, data_val, data_tst = datalib.split(data, config.DATASEED)

data_trn_y = torch.tensor([x.label for x in data_trn]).to(config.DEVICE)
data_val_y = torch.tensor([x.label for x in data_val]).to(config.DEVICE)
data_tst_y = torch.tensor([x.label for x in data_tst]).to(config.DEVICE)

net = Net(meta).to(config.DEVICE)
print(net)

log_file = open(config.LOG_FILE, 'w')

def eval(x, y, name):
	y_ = net.predict(x)
	acc = torch.sum(y == y_).item() / len(x)

	print(f"Metrics ({name}):")
	print(metrics.classification_report(y, y_))

	return acc

def decay_exp(step, start, min, factor, rate):
	exp = step / rate
	value = (start - min) * (factor ** exp) + min

	return value

fps = utils.Fps()
fps.start()

best_acc = 0
fails = 0
lr = config.OPT_LR

wandb.init(project="rcwcf-hmil", name=f'{args.dataset}', config=config)

print("\nTraining...")
for epoch in range(config.TRAINING_EPOCHS):
	batch = random.choices(data_trn, k=config.BATCH_SIZE)
	loss = net.train(batch)

	if utils.is_time(epoch, config.EPOCH_STEPS):
		with torch.no_grad():
			fps_ = fps.fps(epoch)

			trn_acc = eval(data_trn, data_trn_y, 'train')
			val_acc = eval(data_val, data_val_y, 'validation')
			tst_acc = eval(data_tst, data_tst_y, 'test')

			# check for improvement
			if trn_acc > best_acc:
				fails = 0
				best_acc = trn_acc
				impr = "<"
			else:		
				impr = "."

				fails += 1
				if fails >= 5:
					fails = 0
					# lr = decay_exp(epoch, config.OPT_LR, config.OPT_LR_MIN, config.OPT_LR_FACTOR, config.LR_SCHEDULE)
					lr *= config.OPT_LR_FACTOR

					print("\nNew LR:", lr)
					net.set_lr(lr)

					impr = "o"

			print()
			print(f"Epoch {epoch}: acc (trn/val/tst): {trn_acc:.4f}/{val_acc:.4f}/*, FPS: {fps_:.2f} {impr}")
			print(f"{trn_acc:.4f} {val_acc:.4f} {tst_acc:.4f}", file=log_file, flush=True)

			log = {
				'trn_acc': trn_acc,
				'val_acc': val_acc,
				'tst_acc': tst_acc,

				'rate': fps_,
				'loss': loss,
				# 'grad_norm': grad_norm,
				
				'lr': lr,
			}
			wandb.log(log)

	sys.stdout.write('.')
	sys.stdout.flush()


