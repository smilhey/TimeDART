# Time Series Forecasting

A project for time series forecasting featuring TimeDART (Transformer-based) and TCN models, supporting pretraining, fine-tuning, and visual comparisons.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

---

## Features

- 🚀 **Two Model Architectures**
  - TimeDART: Transformer-based with patching
  - TCN: Temporal Convolutional Network
- 🔄 **Training Modes**
  - Masked pretraining
  - Fine-tuning with MSE/Huber loss

---

## Installation

### 1. Dependencies

```bash
pip install torch pandas numpy matplotlib tqdm requests
```

To download some datasets you can use :

```bash
python dl_data.py
```

## Usage

### Examples

- Pretrain TimeDART:

```bash
python train.py \
  --model_name TimeDART \
  --dataset ETTh1.csv \
  --pretrain \
  --n_epochs 50 \
  --batch_size 32 \
  --device cuda`
```

- Finetuing :

```bash
python train.py \
  --model_name TimeDART \
  --finetune \
  --pretrained_model pretrained/ETTh1_pretrain.pth \
  --n_epochs 30 \
  --finetune_loss huber
```

- Generate predictions and visualise them:

```bash
python test.py \
  --pretrained_model models/ETTh1_pretrain.pth \
  --dataset ETTh1.csv \
  --input_len 336 \
  --pred_len 168
```

- Comparing TCN vs TimeDART :

```bash
python TCN_vs_Transformer.py \
  --pretrained_TimeDART TimeDART_ETTh1.pth \
  --pretrained_TCN TCN_ETTh1.pth \
  --pred_len 720
```
