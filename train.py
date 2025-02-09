import os
import argparse
import pandas as pd
import torch

from models.TimeDART import Model as TimeDART
from data.preprocess import TimeSeriesDataset
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="ETTh1.csv")
parser.add_argument("--pretrain", action="store_true")
parser.add_argument("--finetune", action="store_true")
parser.add_argument("--n_epochs", type=int, default=None)
parser.add_argument("--learning_rate", type=float, default=0.0005)
parser.add_argument("--lr_decay", type=float, default=0.9)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--input_len", type=int, default=336)
parser.add_argument("--dropout", type=float, default=0.1)

parser.add_argument("--time_steps", type=int, default=1000)

parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--n_heads", type=int, default=8)
parser.add_argument("--d_ff", type=int, default=2048)
parser.add_argument("--patch_len", type=int, default=2)

parser.add_argument("--e_layers", type=int, default=2)
parser.add_argument("--d_layers", type=int, default=1)

parser.add_argument("--enc_in", type=int, default=7)
parser.add_argument("--dec_in", type=int, default=7)
parser.add_argument("--c_out", type=int, default=7)
parser.add_argument("--mask_ratio", type=float, default=1.0)

parser.add_argument("--use_norm", action="store_true")

parser.add_argument("--scheduler", type=str, default="cosine")
parser.add_argument("--pred_len", type=int, default=None)

parser.add_argument("--device", type=str, default="cuda")

args = parser.parse_args()



def main():
    assert args.n_epochs is not None, "Number of epochs not provided"

    if args.pred_len is None:
        args.pred_len = args.input_len

    args.head_dropout = args.dropout

    assert os.path.exists(f"data/{args.dataset}"), "Dataset not found"
    df = pd.read_csv(f"data/{args.dataset}")
    df.set_index("date", inplace=True)

    data = torch.tensor(df.values).float()
    dataset = TimeSeriesDataset(data, args.input_len, args.pred_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    assert not (args.pretrain and args.finetune), "Choose either pretraining or fine-tuning"
    assert any ([args.pretrain, args.finetune]), "Choose either pretraining or fine-tuning"
    args.task_name = "pretrain" if args.pretrain else "finetune"
    model = TimeDART(args)
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    print('here')
    if args.pretrain:
        print('here2')
        for epoch in range(args.n_epochs):
            model.train()
            train_loss = []
            for x, y in tqdm(dataloader):
                optimizer.zero_grad()
                x = x.to(args.device)
                pred_x = model(x)
                loss = criterion(pred_x, x)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            train_loss = torch.tensor(train_loss).mean()         
            print(f"Epoch: {epoch}, Loss: {train_loss}")
    
    if args.finetune:
        for epoch in range(args.n_epochs):
            model.train()
            train_loss = []
            for x, y in tqdm(dataloader):
                x = x.to(args.device)
                y = y.to(args.device)
                pred_y = model(x)
                loss = criterion(pred_y, y)
                loss.backward()
                train_loss.append(loss.item())
            train_loss = torch.mean(torch.tensor(train_loss))
            print(f"Epoch: {epoch}, Loss: {train_loss}")

if __name__ == "__main__":
    main()