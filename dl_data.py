import os
import requests



def main():
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
    


if __name__ == "__main__":
    main()
