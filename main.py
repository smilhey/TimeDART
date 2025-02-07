import argparse
import os
import requests

parser = argparse.ArgumentParser(
    description="Generate example data, train, infer, visualize results"
)

parser.add_argument("--setup", action="store_true", help="Download experimental data")
parser.add_argument("--train", action="store_true", help="Train the model")
parser.add_argument("--test", action="store_true", help="Visualize the results")


def dl_data():
    datasets = {
        "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
        "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
    }
    os.makedirs("ETT_data", exist_ok=True)

    for name, url in datasets.items():
        response = requests.get(url)
        if response.status_code == 200:
            with open(f"ETT_data/{name}.csv", "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {name}.csv")
        else:
            print(f"Failed to download: {name}.csv")

    print("Done!")


def main():
    args = parser.parse_args()

    if args.setup:
        dl_data()


if __name__ == "__main__":
    main()
