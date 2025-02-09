#!/bin/bash

DATASET="ETTh1.csv"
PRETRAINED_MODEL="from_None_ETTh1_pretrain.pth"
INPUT_LEN=336
PRED_LEN=336
BATCH_SIZE=16
DEVICE="cuda"

python ../test.py \
    --dataset "$DATASET" \
    --pretrained_model "$PRETRAINED_MODEL" \
    --batch_size "$BATCH_SIZE" \
    --input_len "$INPUT_LEN" \
    --pred_len "$PRED_LEN" \
    --device "$DEVICE" \
