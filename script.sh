#!/bin/bash
#
#
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16G
export BACKBONE=resnet3d
export LEARNING_RATE=0.000001
export TO_TRAIN=TRUE
export NUM_EPOCHS=100
export BATCH_SIZE=4 
export SAVE_NAME=test
export DELTA=1
export TRAINPATH=/scratch/users/mpike27/CS230/data/train/SoccerNet/Tensors/
export VALPATH=/scratch/users/mpike27/CS230/data/val/SoccerNet/Tensors/
export TESTPATH=/scratch/users/mpike27/CS230/data/test/SoccerNet/Tensors/
export CLIP_SIZE=60
export HIDDEN_SIZE=512
export NUM_CLASSES=18
export MODEL_TYPE=CONV
export NUM_LAYERS=2
python3 train.py $BACKBONE $LEARNING_RATE $TO_TRAIN $NUM_EPOCHS $BATCH_SIZE $SAVE_NAME $DELTA $TRAINPATH $VALPATH $TESTPATH $CLIP_SIZE $HIDDEN_SIZE $NUM_CLASSES $MODEL_TYPE $NUM_LAYERS