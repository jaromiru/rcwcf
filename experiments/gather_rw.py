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
	TRN_REW_COL = 0
	TRN_CST_COL = 1
	TRN_ACC_COL = 2

	VAL_REW_COL = 3
	VAL_CST_COL = 4
	VAL_ACC_COL = 5

	TST_REW_COL = 6
	TST_CST_COL = 7
	TST_ACC_COL = 8

	data = np.loadtxt(file, ndmin=2)
	if len(data) == 0:
		return None

	best_row = data[:, VAL_ACC_COL].argmax()
	return data[best_row, [TRN_CST_COL, TRN_ACC_COL, VAL_CST_COL, VAL_ACC_COL, TST_CST_COL, TST_ACC_COL]]		# best row
	# return data[-1, [VAL_CST_COL, VAL_ACC_COL, TST_CST_COL, TST_ACC_COL]]			# last row

# -----
if args.show:
	fout = sys.stdout
else:
	fout = open("{path}/results".format(path=args.path), "w")

print('#trn_cst trn_acc val_cst val_acc tst_cst tst_acc budget seed', file=fout)

files = sorted(glob.glob(args.path + "/run_*.log"))

for res_file in files:
	# print(res_file)

	l, s = re.match(r".*run_([\.\d]+)_(.+).log", res_file).groups()
	res = load_file(res_file)
	if res is None:
		print(f"Skipping {res_file}")
		continue

	print("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}".format(*res, l, s), file=fout)
