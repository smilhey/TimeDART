import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import argparse


def prepare_data(data, input_len, pred_len, patch_len, train_size=0.6, val_size=0.2, test_size=0.2):
    """
    Prepare data for training, validation, and testing
    """
    train_data, val_data, test_data = train_val_test_split(data, patch_len, train_size, val_size, test_size)
    scaler = StandardScaler()
    train_data = torch.tensor(scaler.fit_transform(train_data)).float()
    val_data = torch.tensor(scaler.transform(val_data)).float()
    test_data = torch.tensor(scaler.transform(test_data)).float()
    train_dataset = TimeSeriesDataset(train_data, input_len, pred_len)
    val_dataset = TimeSeriesDataset(val_data, input_len, pred_len)
    test_dataset = TimeSeriesDataset(test_data, input_len, pred_len)
    return train_dataset, val_dataset, test_dataset

class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        input_len: int,
        pred_len: int,
    ):
        self.data = data
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return self.data.size(0) - (self.input_len + self.pred_len) + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_len]
        y = self.data[idx + self.input_len : idx + self.input_len + self.pred_len]
        return x, y
    

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 1 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == "decay":
        lr_adjust = {epoch: args.learning_rate * (args.lr_decay ** ((epoch - 1) // 1)) }
    elif args.lradj == "step":
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == "exp":
        lr_adjust = {}
        scheduler.step()

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print('Updating learning rate to {}'.format(lr))


def train_val_test_split(data, patch_len, train_size=0.6, val_size=0.2, test_size=0.2):
    assert train_size + val_size + test_size == 1, "Sizes do not add up to 1"
    n = len(data)
    train_end = int(n * train_size)
    train_end = train_end - (train_end+1) % patch_len
    val_end = train_end + int(n * val_size)
    val_end = val_end - (val_end+1) % patch_len
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    return train, val, test

def create_namespace(dict_args):
    class C:
        pass

    c = C()
    parser = argparse.ArgumentParser(description='Process some integers.')
    for k, v in dict_args.items():
        parser.add_argument('--' + k, type=int, default=v)
    parser.parse_args(namespace=c)
    return c

def early_stopping(val_loss, best_val_loss, counter):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
    return best_val_loss, counter