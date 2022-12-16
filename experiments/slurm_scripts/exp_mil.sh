#!/bin/bash
#SBATCH -p cpufast --time=04:00:00
#SBATCH -n1 -c1
#
#SBATCH --array=0-9

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
WORK_DIR=${EXEC_DIR}/${EXP_NAME}/${EXP}_${DATE}_mil
mkdir -p $WORK_DIR

cp ${EXEC_DIR}/agentr_mil/config.py $WORK_DIR/

ml PyTorch/1.9.0-fosscuda-2020b SciPy-bundle/2020.11-fosscuda-2020b scikit-learn/0.24.1-fosscuda-2020b

seed=$SLURM_ARRAY_TASK_ID

WORK_FILE=$WORK_DIR/run_${seed}.out
LOG_FILE=$WORK_DIR/run_${seed}.log

echo srun -o $WORK_FILE -D $WORK_DIR python -u ${EXEC_DIR}/rcwcf/agentr_mil/main.py $EXP -l2 1e0 -eplen $EPLEN -epochs $EPOCHS -batch $BATCH -log $LOG_FILE -device cpu -l2 $L2 
srun -o $WORK_FILE -D $WORK_DIR python -u ${EXEC_DIR}/rcwcf/agentr_mil/main.py $EXP -l2 1e0 -eplen $EPLEN -epochs $EPOCHS -batch $BATCH -log $LOG_FILE -device cpu -l2 $L2 
