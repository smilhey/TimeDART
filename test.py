import os
import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt

from models.TimeDART import Model as TimeDART
from utils import TimeSeriesDataset, train_val_test_split
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ETTh1.csv")
parser.add_argument("--pretrained_model", type=str, required=True)
parser.add_argument("--input_len", type=int, default=336)
parser.add_argument("--pred_len", type=int, default=336)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()


def main():
    assert os.path.exists(f"{PROJECT_ROOT}/data/{args.dataset}"), "Dataset not found"
    df = pd.read_csv(f"{PROJECT_ROOT}/data/{args.dataset}")
    df.set_index("date", inplace=True)

    data = torch.tensor(df.values).float()
    _, _, test_data = train_val_test_split(data, patch_len=2)

    test_dataset = TimeSeriesDataset(test_data, args.input_len, args.pred_len)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    checkpoint = torch.load(
        f"{PROJECT_ROOT}/models/{args.pretrained_model}", weights_only=False
    )
    model_args = checkpoint["model_args"]
    model_args = argparse.Namespace(**checkpoint["model_args"])
    model = TimeDART(model_args)
    print(model_args)
    model = TimeDART(model_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()

    predictions, actuals = [], []
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(args.device), y.to(args.device)
            pred_y = model(x)[:, -args.pred_len :].cpu().numpy()
            predictions.extend(pred_y)
            actuals.extend(y.cpu().numpy())

    predictions = torch.tensor(predictions).reshape(-1)
    actuals = torch.tensor(actuals).reshape(-1)

    plt.figure(figsize=(12, 5))
    plt.plot(actuals, label="Actual", linestyle="dashed")
    plt.plot(predictions, label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Model Predictions vs Actual Data")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
