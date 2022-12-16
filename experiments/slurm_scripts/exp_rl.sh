#!/bin/bash
#SBATCH -p cpulong
#SBATCH -n1 -c1
#
#SBATCH --array=0-29

if [[ $# -ne 8 ]] ; then
    echo 'invalid params'
    exit 1
fi

EXEC_DIR=~/rcwcf

EXP_NAME=$1
EXP=$2
EPLEN=$3
EPOCHS=$4
BATCH=$5
A_H=$6
A_H_MIN=$7
L2=$8

DATE=`date +%d_%m_%Y`
WORK_DIR=${EXEC_DIR}/${EXP_NAME}/${EXP}_${DATE}_rl
mkdir -p $WORK_DIR

cp ${EXEC_DIR}/agentr_rl/config.py $WORK_DIR/

ml PyTorch/1.9.0-fosscuda-2020b SciPy-bundle/2020.11-fosscuda-2020b scikit-learn/0.24.1-fosscuda-2020b

lambdas=(0.0001 0.0001269 0.00016103 0.00020434 0.00025929 0.00032903 0.00041753 0.00052983 0.00067234 0.00085317 0.00108264 0.00137382 0.00174333 0.00221222 0.00280722 0.00356225 0.00452035 0.00573615 0.00727895 0.00923671 0.01172102 0.01487352 0.01887392 0.02395027 0.03039195 0.0385662 0.04893901 0.06210169 0.07880463 0.1)

lambda=${lambdas[$SLURM_ARRAY_TASK_ID]}
seed=$SLURM_ARRAY_TASK_ID

WORK_FILE=$WORK_DIR/run_${lambda}_${seed}.out
MODEL_FILE=$WORK_DIR/run_${lambda}_${seed}.mdl
LOG_FILE=$WORK_DIR/run_${lambda}_${seed}.log

echo srun -o $WORK_FILE -D $WORK_DIR python -u ${EXEC_DIR}/agentr_rl/main.py $EXP $lambda -eplen $EPLEN -epochs $EPOCHS -batch $BATCH -log $LOG_FILE -model $MODEL_FILE -alpha_h $A_H -alpha_h_min $A_H_MIN -l2 $L2
srun -o $WORK_FILE -D $WORK_DIR python -u ${EXEC_DIR}/agentr_rl/main.py $EXP $lambda -eplen $EPLEN -epochs $EPOCHS -batch $BATCH -log $LOG_FILE -model $MODEL_FILE -alpha_h $A_H -alpha_h_min $A_H_MIN -l2 $L2
