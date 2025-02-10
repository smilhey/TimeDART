from tqdm import tqdm
import numpy as np
import os
import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt

from models.TimeDART import Model as TimeDART
from utils import StrideTimeSeriesDataset, load_test_data
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
    num_features = data.shape[1]

    _,_, data = load_test_data(data)
    print(data.shape)


    dataset = StrideTimeSeriesDataset(data, args.pred_len, args.input_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    checkpoint = torch.load(
        f"{PROJECT_ROOT}/models/{args.pretrained_model}", weights_only=False
    )

    model_args = argparse.Namespace(**checkpoint["model_args"])

    model = TimeDART(model_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for x in tqdm((dataloader)):
            x = x.to(args.device)
            pred_x = (
                model(x)[:, -args.pred_len :].cpu().numpy()
            )  # [batch, pred_len, num_features]
            predictions.append(np.concatenate(pred_x, axis=0))

    predictions = np.concatenate(predictions, axis=0)
    print(predictions.shape)
    actuals = data.numpy()
    print(actuals.shape)

    fig, axes = plt.subplots(
        num_features, 1, figsize=(14, 3 * num_features), sharex=True
    )

    if num_features == 1:
        axes = [axes]  # Make it iterable for a single-variable case

    for feature_idx in range(num_features):
        ax = axes[feature_idx]
        ax.plot(
            actuals[:, feature_idx],
            label=f"Actual",
            color="gray",
            alpha=0.5,
        )
        ax.plot(
            range(args.input_len, args.input_len +len(predictions)),
            predictions[:, feature_idx],
            label="Pred",
            alpha=0.8,
        )
        ax.set_ylabel(df.columns[feature_idx])
        
        for i in range(0, len(predictions), args.pred_len):
            ax.axvline(x=args.input_len + i, color="red", linestyle="--", alpha=0.25, label=f"Prediction Boundary ({args.pred_len} steps)")
            if feature_idx == 0 and i == 0:
                ax.legend(draggable=True)
    plt.xlabel("Time")
    plt.suptitle("Multivariate Time Series with Overlaid Predictions")
    plt.show()


if __name__ == "__main__":
    main()
