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

budgets=(1. 1.65517241 2.31034483 2.96551724 3.62068966 4.27586207 4.93103448 5.5862069 6.24137931 6.89655172 7.55172414 8.20689655 8.86206897 9.51724138 10.17241379 10.82758621 11.48275862 12.13793103 12.79310345 13.44827586 14.10344828 14.75862069 15.4137931 16.06896552 16.72413793 17.37931034 18.03448276 18.68965517 19.34482759 20.)

budget=${budgets[$SLURM_ARRAY_TASK_ID]}
seed=$SLURM_ARRAY_TASK_ID

WORK_FILE=$WORK_DIR/run_${budget}_${seed}.out
MODEL_FILE=$WORK_DIR/run_${budget}_${seed}.mdl
LOG_FILE=$WORK_DIR/run_${budget}_${seed}.log

echo srun -o $WORK_FILE -D $WORK_DIR python -u ${EXEC_DIR}/agentr_rw/main.py $EXP $budget -eplen $EPLEN -epochs $EPOCHS -batch $BATCH -log $LOG_FILE -model $MODEL_FILE -l2 $L2 
srun -o $WORK_FILE -D $WORK_DIR python -u ${EXEC_DIR}/agentr_rw/main.py $EXP $budget -eplen $EPLEN -epochs $EPOCHS -batch $BATCH -log $LOG_FILE -model $MODEL_FILE -l2 $L2 
