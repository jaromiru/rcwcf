import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.legend_handler import HandlerTuple

import seaborn as sns

from cycler import cycler

import numpy as np, pandas as pd, re, json
import argparse

# from loess.loess_1d import loess_1d
from moepy import lowess
# from statsmodels.nonparametric.smoothers_lowess import lowess
# import scipy.interpolate
from scipy.signal import savgol_filter

# ---- arguments
parser = argparse.ArgumentParser()
parser.add_argument('sources', nargs='+', help="sources to show")
parser.add_argument('-dest', help="save to file")

parser.add_argument('-styles', nargs='+', help="List of styles to apply (k+, b-/+, etc.)")
parser.add_argument('-labels', nargs='+', help="List of labels")
parser.add_argument('-nolegend',  action='store_const', const=True, help="Supress the legend")
parser.add_argument('-noylabel',  action='store_const', const=True, help="Supress the ylabel")

parser.add_argument('-val',  action='store_const', const=True, help="display only validation data")
parser.add_argument('-trn',  action='store_const', const=True, help="display only training data")

parser.add_argument('-ylim', help="y-lim")
parser.add_argument('-xlim', help="x-lim")

args = parser.parse_args()

# ----
def pareto_front(data):
	sort_idx = np.argsort(data[:, 0])
	data = data[sort_idx]

	pareto = [0]
	for idx, it in enumerate(data[1:], 1):
		if it[1] >= data[pareto[-1], 1]:
			pareto.append(idx)

	return sort_idx[pareto], data[pareto]

def best_validation(data):
	best_row = np.argmax(data[:, 0])
	return data[best_row]

# sns.set()
# sns.set(style="darkgrid", font_scale=1.5)

# ---- plot
mpl.rc('font',size=40)
# mpl.rc('text',usetex=True)
mpl.rc('text', usetex=True)
mpl.rc('font', family='sans-serif')
# plt.rcParams['pdf.fonttype'] = 42
mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

fig = plt.figure(figsize=(8, 8))
ax = plt.axes([0.195, 0.155, 0.80, 0.8])
# print(ax)

def str_endswith(str, ends):
	for e in ends:
		if str.endswith(e):
			return True

	return False

# using bootstrap method from https://james-brennan.github.io/posts/lowess_conf/
# def smooth(x, y, x_grid, frac=1/2):
# 	samples = np.random.choice(len(x), int( len(x) * frac ), replace=False)

# 	x = x[samples]
# 	y = y[samples]

# 	y_lowess = lowess(y, x, frac=2/3, return_sorted=False)
# 	y_grid = scipy.interpolate.interp1d(x, y_lowess, fill_value='extrapolate')(xgrid)

# 	return y_grid

# ----------------- preprare metrics for AUC
dataset = args.sources[0].split('/')[-1].rsplit( '_', 1)[0]
class_dists = json.load(open('../data_conversions/class_count.json'))
class_dist = [x['classes'] for x in class_dists if x['dataset'] == dataset][0]
prior_acc = max([x['count'][1] for x in class_dist])

hmil_perfs = [x.split(' ') for x in open('hmil_perf.txt').read().splitlines()]
hmil_perf = [x for x in hmil_perfs if x[0] == dataset][0]
hmil_cst = float(hmil_perf[1])
hmil_acc = float(hmil_perf[2])
# -----------------

legend_list = []
legend_labels = []
data_max_x = 0
for src_idx, src in enumerate(args.sources):
	algo = src.split('_')[-1]
	data = np.loadtxt(f'{src}/results', ndmin=2)

	if args.styles:
		style_color, style_line, style_marker = args.styles[src_idx].split('/')
	else:
		style_color, style_line, style_marker = None, None, None

	if args.labels:
		label = args.labels[src_idx]
	else:
		label = src.replace('-', ' ').replace('_', '-')

	# special case, different format
	# if str_endswith(src, ["_mil", "_mil/", "_best", "_best/"]):
	if str_endswith(src, ["_mil", "_mil/"]):
		# data = best_validation(data)

		if args.trn:
			x = data[:, 0]
			y = data[:, 1]

		elif args.val:
			x = data[:, 2]
			y = data[:, 3]

		else:
			x = data[:, 4]
			y = data[:, 5]

		x_max = x[0]
		y_std = np.std(y)
		y_avg = np.mean(y)

		data_max_x = max(data_max_x, x_max)
		
		ax.plot([0, data_max_x * 1.1], [y_avg, y_avg], color=style_color, linestyle=style_line, alpha=0.5, linewidth=1.0, clip_on=False, in_layout=False)
		ax.fill_between([0, data_max_x * 1.1], y_avg+y_std, y_avg-y_std, color=style_color, alpha=0.1, edgecolor='none')

		p2 = ax.errorbar(x_max, y_avg, yerr=y_std, marker="_", capsize=3, linewidth=1, color=style_color, linestyle='None', clip_on=False)
		p1 = ax.scatter(x, y, marker=style_marker, color=style_color, clip_on=False, alpha=0.3)
		# plt.plot([data_max_x], [y_avg], marker=style_marker, color=style_color, clip_on=False, alpha=1.0, linestyle='None', label=label)

		legend_list.append((p1, p2))
		legend_labels.append(label)

		# print HMIL cost / acc
		# dataset = src.split('/')[-1].rsplit( '_', 1)[0]
		# print(f'{dataset} {x_max} {y_avg}')

		# if args.styles:
		# 	plt.plot([0, data_max_x * 1.1], [data[1], data[1]], color=style_color, linestyle=style_line, linewidth=1.0, in_layout=False, label=label)
		# 	plt.annotate(f"acc\\texttt{{=}}{data[1]:.2f}; cost\\texttt{{=}}{data[0]:.1f}", xy=(data_max_x, data[1]), xytext=(-192,-30), textcoords='offset pixels', color=style_color, fontsize='x-small', verticalalignment='bottom')
		# 	# plt.scatter(data[0], data[1], marker=style_marker, color=style_color, linewidth=5, clip_on=False, label=label)
		# else:
		# 	plt.scatter(data[0], data[1], data, marker="v", linewidth=5, clip_on=False, label=label)

		auc = (hmil_acc - prior_acc) / 2. * hmil_cst
		auc_norm = auc / hmil_cst / (1 - prior_acc)

		print(f'{dataset}-{algo}: {auc=:.2f} {auc_norm=:.2f}')

	else:
		data_org = data
		data_trn = data[:, :2]
		data_val = data[:, 2:4]
		data_tst = data[:, 4:6]

		if args.trn:
			data = data[:, :2]
		
		elif args.val:
			data = data[:, 2:4]
		
		else:
			data = data[:, 4:6]


		x = data[:, 0]
		y = data[:, 1]

		data_max_x = max(data_max_x, np.max(x))

		# PARETO
		pareto_idx, _ = pareto_front(data_val)
		data_pareto = data[pareto_idx]
		data_pareto = data_pareto[np.argsort(data_pareto[:, 0])]
		# print(data_pareto)

		pareto_idx_tst, _ = pareto_front(data_pareto)
		data_pareto_tst = data_pareto[pareto_idx_tst]
		data_pareto_tst = data_pareto_tst[np.argsort(data_pareto_tst[:, 0])]

		# plt.plot(data_pareto[:, 0], data_pareto[:, 1], marker=None, color=style_color, linestyle=style_line, linewidth=1, label=label, clip_on=False)
		p2, = plt.plot(data_pareto_tst[:, 0], data_pareto_tst[:, 1], marker=style_marker, markeredgecolor="None", color=style_color, linestyle=style_line, linewidth=1, label=label, clip_on=False)

		_ = ax.scatter(x, y, marker=style_marker, edgecolors="None", color=style_color, linewidth=2, clip_on=False, alpha=0.1)
		p1 = ax.scatter(data_pareto[:, 0], data_pareto[:, 1], marker=style_marker, edgecolors="None", color=style_color, linewidth=2, clip_on=False, alpha=1.0)

		# plot mean lowess and its std
		# xgrid = np.linspace(x.min(),x.max())
		# smooths = np.stack([smooth(x, y, xgrid, frac=0.5) for i in range(100)]).T
		# plt.plot(xgrid, smooths, color='tomato', alpha=0.25)

		# mean = np.nanmean(smooths, axis=1)
		# std = np.nanstd(smooths, axis=1)

		# plt.fill_between(xgrid, mean+std, mean-std, color=style_color, alpha=0.15, edgecolor='none')
		# plt.plot(xgrid, mean, marker=None, color=style_color, linestyle=style_line, linewidth=3, label=label, clip_on=False)

		# interp = lowess(y, x, frac=0.5, it=5)
		# plt.plot(interp[:, 0], interp[:, 1], marker=None, color=style_color, linestyle=style_line, linewidth=3, label=label, clip_on=False)


		# x_order = np.argsort(x)
		# x_ord = x[x_order]
		# y_ord = y[x_order]
		# x_new, y_new = savgol_filter((x_ord, y_ord), 11, 3)
		# plt.plot(x_new, y_new, marker=None, color=style_color, linestyle=style_line, linewidth=3, label=label, clip_on=False)

		# LOWESS
		lowess_model = lowess.Lowess()
		# lowess_model.fit(x, y, frac=0.8, num_fits=5, robust_iters=3)
		lowess_model.fit(data_pareto[:, 0], data_pareto[:, 1], frac=0.8, num_fits=20, robust_iters=10)

		# Model prediction
		x_new = np.sort(data_pareto[:, 0])
		y_new = lowess_model.predict(x_new)

		# p2, = ax.plot(x_new, y_new, marker=None, color=style_color, linestyle=style_line, linewidth=3)

		legend_list.append(p2)
		legend_labels.append(label)

		# AUC
		x_auc = np.concatenate([[0.], data_pareto_tst[:, 0], [hmil_cst]])
		y_auc = np.concatenate([[prior_acc], data_pareto_tst[:, 1], [hmil_acc]])

		y_auc = np.fmin(y_auc, 1.0)
		y_auc = np.fmax(y_auc, prior_acc)

		# plt.plot(x_auc, y_auc, marker=None, color='k', linestyle=style_line, linewidth=1)		
		# plt.fill_between(x_auc, prior_acc, y_auc, color=style_color, alpha=0.1, edgecolor='none')

		auc = np.trapz(y_auc - prior_acc, x_auc)  # remove the bottom part
		auc_norm = auc / hmil_cst / (1 - prior_acc)

		print(f'{dataset}-{algo}: {auc=:.2f} {auc_norm=:.2f}')

if not args.nolegend:
	ax.legend(legend_list, legend_labels, handler_map={tuple: HandlerTuple(ndivide=None)}, frameon=True, loc='lower right', prop={'size': 24})
	# plt.legend(frameon=True, loc='lower right', prop={'size': 24})

plt.xlabel("cost")

if not args.noylabel:
	plt.ylabel("accuracy")

if args.ylim:
	plt.ylim(*np.fromstring(args.ylim, sep=','))

if args.xlim:
	plt.xlim(*np.fromstring(args.xlim, sep=','))
else:
	plt.xlim(0, data_max_x * 1.1)

plt.minorticks_on()
plt.tick_params(which='major', width=1, length=10)
plt.tick_params(which='minor', width=1, length=3)

plt.ioff()
if args.dest:
	plt.savefig(args.dest, dpi=fig.dpi)
else:
	plt.show()
