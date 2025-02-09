import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        input_len: int,
        pred_len: int,
    ):
        self.data = data
        scaler = StandardScaler()
        self.data = torch.tensor(scaler.fit_transform(self.data)).float()
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return self.data.size(0) - (self.input_len + self.pred_len) + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_len]
        y = self.data[idx + self.input_len : idx + self.input_len + self.pred_len]
        return x, y