#!/bin/bash

DATASET="ETTh1.csv"
N_EPOCHS=1
LEARNING_RATE=0.0005
BATCH_SIZE=16
INPUT_LEN=336
DROPOUT=0.1
DEVICE="cuda"
NUM_WORKERS=1

python ../train.py \
    --dataset "$DATASET" \
    --pretrain \
    --n_epochs "$N_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --batch_size "$BATCH_SIZE" \
    --input_len "$INPUT_LEN" \
    --dropout "$DROPOUT" \
    --device "$DEVICE" \
    --num_workers "$NUM_WORKERS"
