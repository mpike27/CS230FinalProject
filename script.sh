# script.sh
export BACKBONE=resnet3d
export LEARNING_RATE=0.0001
export TO_TRAIN=TRUE
export NUM_EPOCHS=50
export BATCH_SIZE=4 
export SAVE_NAME=test
export DELTA=1
export TRAINPATH=train
export VALPATH=val
export TESTPATH=test
python3 train.py $BACKBONE $LEARNING_RATE $TO_TRAIN $NUM_EPOCHS $BATCH_SIZE