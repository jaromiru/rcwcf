import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import argparse, os, re

parser = argparse.ArgumentParser()
parser.add_argument('source', help="source debug")
parser.add_argument('-title', help="figure title")
parser.add_argument('-dest', help="save to file")
args = parser.parse_args()

data = np.loadtxt(args.source, ndmin=2)
# data = data[:50]

trn_rew = data[:, 0]; trn_cst = data[:, 1]; trn_acc = data[:, 2]
val_rew = data[:, 3]; val_cst = data[:, 4]; val_acc = data[:, 5]
tst_rew = data[:, 6]; tst_cst = data[:, 7]; tst_acc = data[:, 8]

mpl.rc('font',size=13)
# mpl.rc('text',usetex=True)
fig, ax = plt.subplots(3,1,figsize=(8,5.5))

trn_style = 'b--'
val_style = 'k-'
tst_style = 'k-'

ax[0].plot(trn_rew, trn_style, linewidth=2, label='training data')
ax[1].plot(trn_cst, trn_style, linewidth=2)
ax[2].plot(trn_acc, trn_style, linewidth=2)

ax[0].plot(val_rew, val_style, linewidth=2, label='validation data')
ax[1].plot(val_cst, val_style, linewidth=2)
ax[2].plot(val_acc, val_style, linewidth=2)

ax[0].plot(tst_rew, tst_style, alpha=0.5, linewidth=1, label='testing data')
ax[1].plot(tst_cst, tst_style, alpha=0.5, linewidth=1)
ax[2].plot(tst_acc, tst_style, alpha=0.5, linewidth=1)

for i, title in enumerate(['reward', 'cost', 'accuracy']):
	ax[i].set_ylabel(title)
	ax[i].minorticks_on()
	ax[i].set_xlim(0, len(data)-1)

	if title != 'reward':
		ax[i].set_ylim(0)
	else:
		if (not np.any(trn_rew < 0)) and (not np.any(val_rew < 0)):
			ax[i].set_ylim(0)

rew_ylim = ax[0].get_ylim()

# plot the maximum indicator
max_idx = val_rew.argmax()
ax[0].plot([max_idx, max_idx], [rew_ylim[0], val_rew[max_idx]], 'r-.', linewidth=2, label="best iteration")
ax[1].plot([max_idx, max_idx], [0, val_cst[max_idx]], 'r-.', linewidth=2)
ax[2].plot([max_idx, max_idx], [0, val_acc[max_idx]], 'r-.', linewidth=2)

ax[0].set_ylim(rew_ylim)

ax[0].legend(frameon=False, loc='lower right', fontsize='small')

plt.xlabel("epochs")

datasets = {'carc_rl': 'carcinogenesis',
			'toy_b_rl': 'synthetic',
			'hepa_rl': 'hepatitis',
			'muta_rl': 'mutagenesis',
			'recipes_rl': 'ingredients',
			'sap_balanced_rl': 'sap',
			'stats_rl': 'stats',
			'stats_full_rl': 'stats',
			'web_100k_rl': 'threatcrowd'}

dt, lmb, seed = re.match("(.+)/run_(.+)_(.+)", args.title).groups()
# fig.suptitle(f"{datasets[dt]}, $\\lambda$={lmb}, seed={seed}")
ax[0].title.set_text(f"{datasets[dt]}, $\\lambda$={lmb}")

ax[0].xaxis.set_ticklabels([])
ax[1].xaxis.set_ticklabels([])

if args.dest:

	dst = args.dest + f"/{dt}_{lmb.replace('.', '-')}_{seed}.pdf"
	os.makedirs(args.dest, exist_ok=True)
	# os.makedirs(os.path.dirname(args.dest), exist_ok=True)
	plt.savefig(dst, bbox_inches='tight')
else:
	plt.show()
