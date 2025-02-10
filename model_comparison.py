"""For different TimeDARTs comparison"""

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
parser.add_argument("--model_1", type=str, required=True)
parser.add_argument("--model_2", type=str, required=True)
parser.add_argument("--name_1", type=str, default="Classic TimeDART")
parser.add_argument("--name_2", type=str, default="Random")
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

    checkpoint_1 = torch.load(
        f"{PROJECT_ROOT}/models/{args.model_1}", weights_only=False
    )

    model_1_args = argparse.Namespace(**checkpoint_1["model_args"])

    model_1 = TimeDART(model_1_args)
    model_1.load_state_dict(checkpoint_1["model_state_dict"])
    model_1.to(args.device)
    model_1.eval()

    checkpoint_2 = torch.load(
        f"{PROJECT_ROOT}/models/{args.model_2}", weights_only=False
    )

    model_2_args = argparse.Namespace(**checkpoint_2["model_args"])
    model_2 = TimeDART(model_2_args)
    model_2.load_state_dict(checkpoint_2["model_state_dict"])
    model_2.to(args.device)
    model_2.eval()

    predictions_1 = []
    predictions_2 = []

    with torch.no_grad():
        for x in tqdm((dataloader)):
            x = x.to(args.device)
            pred_x_1 = (
                model_1(x)[:, -args.pred_len :].cpu().numpy()
            )  # [batch, pred_len, num_features]
            pred_x_2 = (
                model_2(x)[:, -args.pred_len :].cpu().numpy()
            )  # [batch, pred_len, num_features]
            predictions_1.append(np.concatenate(pred_x_1, axis=0))
            predictions_2.append(np.concatenate(pred_x_2, axis=0))

    predictions_1 = np.concatenate(predictions_1, axis=0)
    predictions_2 = np.concatenate(predictions_2, axis=0)
    # print(predictions.shape)
    actuals = data.numpy()
    # print(actuals.shape)

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
            range(args.input_len, args.input_len +len(predictions_1)),
            predictions_1[:, feature_idx],
            label=f"Pred ({args.name_1})",
            alpha=0.8,
        )
        ax.plot(
            range(args.input_len, args.input_len +len(predictions_2)),
            predictions_2[:, feature_idx],
            label=f"Pred ({args.name_2})",
            alpha=0.8,
        )
        ax.set_ylabel(df.columns[feature_idx])
        for i in range(0, len(predictions_1), args.pred_len):
            ax.axvline(x=args.input_len + i, color="red", linestyle="--", alpha=0.25, label=f"Prediction Boundary ({args.pred_len} steps)")
            if feature_idx == 0 and i == 0:
                ax.legend(draggable=True)

    plt.xlabel("Time")
    plt.suptitle("Multivariate Time Series with Overlaid Predictions - Random init vs pretrain")
    plt.show()


if __name__ == "__main__":
    main()
