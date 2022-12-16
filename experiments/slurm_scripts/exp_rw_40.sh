#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH -n1 -c1
#
#SBATCH --array=0-29

if [[ $# -ne 6 ]] ; then
    echo 'invalid params'
    exit 1
fi

EXEC_DIR=~/rcwcf

EXP_NAME=$1
EXP=$2
EPLEN=$3
EPOCHS=$4
BATCH=$5
L2=$6

DATE=`date +%d_%m_%Y`
WORK_DIR=${EXEC_DIR}/${EXP_NAME}/${EXP}_${DATE}_rw
mkdir -p $WORK_DIR

cp ${EXEC_DIR}/agentr_rw/config.py $WORK_DIR/

ml PyTorch/1.9.0-fosscuda-2020b SciPy-bundle/2020.11-fosscuda-2020b scikit-learn/0.24.1-fosscuda-2020b

budgets=(1. 2.34482759 3.68965517 5.03448276 6.37931034 7.72413793 9.06896552 10.4137931 11.75862069 13.10344828 14.44827586 15.79310345 17.13793103 18.48275862 19.82758621 21.17241379 22.51724138 23.86206897 25.20689655 26.55172414 27.89655172 29.24137931 30.5862069 31.93103448 33.27586207 34.62068966 35.96551724 37.31034483 38.65517241 40.)

budget=${budgets[$SLURM_ARRAY_TASK_ID]}
seed=$SLURM_ARRAY_TASK_ID

WORK_FILE=$WORK_DIR/run_${budget}_${seed}.out
MODEL_FILE=$WORK_DIR/run_${budget}_${seed}.mdl
LOG_FILE=$WORK_DIR/run_${budget}_${seed}.log

echo srun -o $WORK_FILE -D $WORK_DIR python -u ${EXEC_DIR}/agentr_rw/main.py $EXP $budget -eplen $EPLEN -epochs $EPOCHS -batch $BATCH -log $LOG_FILE -model $MODEL_FILE -l2 $L2
srun -o $WORK_FILE -D $WORK_DIR python -u ${EXEC_DIR}/agentr_rw/main.py $EXP $budget -eplen $EPLEN -epochs $EPOCHS -batch $BATCH -log $LOG_FILE -model $MODEL_FILE -l2 $L2
