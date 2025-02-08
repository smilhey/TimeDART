import math
import torch
import torch.nn as nn


class PosEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.device = device

    def forward(self, x):
        pe = torch.zeros(self.max_len, self.d_model).float()
        pe.requires_grad = False
        position = torch.arange(0, self.max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, self.d_model, 2).float()
            * -(math.log(10000.0) / self.d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe[:, : x.size(1)].to(self.device)


class PosEncoding(nn.Module):
    def __init__(self, d_model, dropout, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.pos_embedding = PosEmbedding(d_model, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pos_embedding(x))


class Patch(nn.Module):
    def __init__(self, stride, patch_len):
        super().__init__()
        self.stride = stride
        self.patch_len = patch_len

    def forward(self, x):
        x = x.squeeze(-1)
        x = x.unfold(-1, self.patch_len, self.stride)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, d_model):
        super().__init__()
        self.embedding = nn.Linear(patch_len, d_model)

    def forward(self, x):
        return self.embedding(x)


class ChannelIndependence(nn.Module):
    def __init__(
        self,
        input_len: int,
    ):
        super().__init__()
        self.input_len = input_len

    def forward(self, x):
        """
        :param x: [batch_size, input_len, num_features]
        :return: [batch_size * num_features, input_len, 1]
        """
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, self.input_len, 1)
        return x


class AddSosTokenAndDropLast(nn.Module):
    def __init__(self, sos_token: torch.Tensor):
        super().__init__()
        assert sos_token.dim() == 3
        self.sos_token = sos_token

    def forward(self, x):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        sos_token_expanded = self.sos_token.expand(
            x.size(0), -1, -1
        )  # [batch_size * num_features, 1, d_model]
        x = torch.cat(
            [sos_token_expanded, x], dim=1
        )  # [batch_size * num_features, seq_len + 1, d_model]
        x = x[:, :-1, :]  # [batch_size * num_features, seq_len, d_model]
        return x
