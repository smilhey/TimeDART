import torch.optim.lr_scheduler as lr_scheduler

import os
import argparse
import pandas as pd
import torch

from models.TimeDART import Model as TimeDART
from utils.split import train_val_test_split
from utils.preprocess import TimeSeriesDataset
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

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

parser.add_argument("--d_model", type=int, default=32)
parser.add_argument("--n_heads", type=int, default=16)
parser.add_argument("--d_ff", type=int, default=64)
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
parser.add_argument("--num_workers", type=int, default=1)

parser.add_argument("--pretrained_model", type=str, default=None)

args = parser.parse_args()


def main():
    assert args.n_epochs is not None, "Number of epochs not provided"

    if args.pred_len is None:
        args.pred_len = args.input_len

    args.head_dropout = args.dropout

    torch.cuda.empty_cache()

    assert os.path.exists(f"{PROJECT_ROOT}/data/{args.dataset}"), "Dataset not found"
    df = pd.read_csv(f"{PROJECT_ROOT}/data/{args.dataset}")
    df.set_index("date", inplace=True)

    data = torch.tensor(df.values).float()
    train_data, val_data, test_data = train_val_test_split(data, args.patch_len)

    train_dataset = TimeSeriesDataset(train_data, args.input_len, args.pred_len)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_dataset = TimeSeriesDataset(val_data, args.input_len, args.pred_len)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_dataset = TimeSeriesDataset(test_data, args.input_len, args.pred_len)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    assert not (
        args.pretrain and args.finetune
    ), "Choose either pretraining or fine-tuning"
    assert any(
        [args.pretrain, args.finetune]
    ), "Choose either pretraining or fine-tuning"
    args.task_name = "pretrain" if args.pretrain else "finetune"
    model = TimeDART(args)
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    if args.scheduler == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
    elif args.scheduler == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.lr_decay)
    else:
        scheduler = None

    if args.pretrain:
        print("Pretraining : ")
        for epoch in range(args.n_epochs):
            model.train()
            train_loss = []
            for x, y in tqdm(train_dataloader):
                optimizer.zero_grad()
                x = x.to(args.device)
                pred_x = model(x)
                loss = criterion(pred_x, x)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            if scheduler:
                scheduler.step()

            train_loss = torch.tensor(train_loss).mean()
            print(
                f"Epoch: {epoch}, Loss: {train_loss}, LR: {optimizer.param_groups[0]['lr']}"
            )

            if epoch % 5 == 4:
                model.eval()
                val_loss = []
                with torch.no_grad():
                    for x, y in val_dataloader:
                        x = x.to(args.device)
                        pred_x = model(x)
                        loss = criterion(pred_x, x)
                        val_loss.append(loss.item())
                    val_loss = torch.tensor(val_loss).mean()
                    print(f"Val Loss: {val_loss}")

        model.eval()
        test_loss = []
        with torch.no_grad():
            for x, y in test_dataloader:
                x = x.to(args.device)
                pred_x = model(x)
                loss = criterion(pred_x, x)
                test_loss.append(loss.item())
            test_loss = torch.tensor(test_loss).mean()
            print(f"Test Loss: {test_loss}")

        torch.save(
            model.state_dict(),
            f"{PROJECT_ROOT}/models/{args.dataset.split('.')[0]}_{args.task_name}.pth",
        )

    if args.finetune:
        assert args.pretrained_model is not None, "Pretrained model not provided"
        state = torch.load(f"models/{args.pretrained_model}")
        # print(state.keys())
        # print(model.state_dict().keys())
        for key in list(model.state_dict().keys()):
            if key not in state.keys():
                state[key] = model.state_dict()[key].clone()
        for key in list(state.keys()):
            if key not in model.state_dict().keys():
                del state[key]
        model.load_state_dict(state)
        print("Finetuning : ")
        for epoch in range(args.n_epochs):
            model.train()
            train_loss = []
            for x, y in tqdm(train_dataloader):
                optimizer.zero_grad()
                x = x.to(args.device)
                y = y.to(args.device)
                pred_y = model(x)[:, -args.pred_len :]
                loss = criterion(pred_y, y)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            if scheduler:
                scheduler.step()

            train_loss = torch.tensor(train_loss).mean()
            print(
                f"Epoch: {epoch}, Loss: {train_loss}, LR: {optimizer.param_groups[0]['lr']}"
            )

            model.eval()
            val_loss = []
            with torch.no_grad():
                for x, y in val_dataloader:
                    x = x.to(args.device)
                    y = y.to(args.device)
                    pred_y = model(x)[:, -args.pred_len :]
                    loss = criterion(pred_y, y)
                    val_loss.append(loss.item())
                val_loss = torch.mean(torch.tensor(val_loss))
                print(f"Val Loss: {val_loss}")

        model.eval()
        test_loss = []
        with torch.no_grad():
            for x, y in test_dataloader:
                x = x.to(args.device)
                y = y.to(args.device)
                pred_y = model(x)[:, -args.pred_len :]
                loss = criterion(pred_y, y)
                test_loss.append(loss.item())
            test_loss = torch.mean(torch.tensor(test_loss))
            print(f"Test Loss: {test_loss}")

    torch.save(
        model.state_dict(),
        f"{PROJECT_ROOT}/models/from_{args.pretrained_model}_{args.dataset.split('.')[0]}_{args.task_name}.pth",
    )


if __name__ == "__main__":
    main()
