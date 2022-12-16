# ls -d exp-*/ | xargs -n 1 python3.6 gather_raw.py

import numpy as np
import sys, glob, re, os, subprocess
import argparse

# -----
parser = argparse.ArgumentParser()
parser.add_argument('path', help="path to process")
parser.add_argument('--show', action='store_const', const=True, help="display on screen (default save)")

args = parser.parse_args()

# -----
def load_file(file):
	TRN_ACC_COL = 0
	VAL_ACC_COL = 1
	TST_ACC_COL = 2

	TRN_CST_COL = 0
	VAL_CST_COL = 1
	TST_CST_COL = 2

	data = np.loadtxt(file, ndmin=2)

	best_row = data[:, VAL_ACC_COL].argmax()
	dbest = data[best_row]

	return [costs[TRN_CST_COL], dbest[TRN_ACC_COL], costs[VAL_CST_COL], dbest[VAL_ACC_COL], costs[TST_CST_COL], dbest[TST_ACC_COL]]

# -----
if args.show:
	fout = sys.stdout
else:
	fout = open("{path}/results".format(path=args.path), "w")

dataset = re.match(r".*/(.+)_mil", args.path).group(1)
costs = np.loadtxt(f'hmil_costs/{dataset}')

print('#trn_cst trn_acc val_cst val_acc tst_cst tst_acc seed', file=fout)
files = sorted(glob.glob(args.path + "/run_*.log"))
for res_file in files:

	s = re.match(r".*run_(\d+).log", res_file).group(1)
	res = load_file(res_file)

	print("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(*res, s), file=fout)
