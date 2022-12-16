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

budgets=(1. 1.31034483 1.62068966 1.93103448 2.24137931 2.55172414 2.86206897 3.17241379 3.48275862 3.79310345 4.10344828 4.4137931 4.72413793 5.03448276 5.34482759 5.65517241 5.96551724 6.27586207 6.5862069 6.89655172 7.20689655 7.51724138 7.82758621 8.13793103 8.44827586 8.75862069 9.06896552 9.37931034 9.68965517 10.)

budget=${budgets[$SLURM_ARRAY_TASK_ID]}
seed=$SLURM_ARRAY_TASK_ID

WORK_FILE=$WORK_DIR/run_${budget}_${seed}.out
MODEL_FILE=$WORK_DIR/run_${budget}_${seed}.mdl
LOG_FILE=$WORK_DIR/run_${budget}_${seed}.log

echo srun -o $WORK_FILE -D $WORK_DIR python -u ${EXEC_DIR}/agentr_rw/main.py $EXP $budget -eplen $EPLEN -epochs $EPOCHS -batch $BATCH -log $LOG_FILE -model $MODEL_FILE -l2 $L2
srun -o $WORK_FILE -D $WORK_DIR python -u ${EXEC_DIR}/agentr_rw/main.py $EXP $budget -eplen $EPLEN -epochs $EPOCHS -batch $BATCH -log $LOG_FILE -model $MODEL_FILE -l2 $L2
