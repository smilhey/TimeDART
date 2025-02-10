import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D Causal Convolution with Padding"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation  # Causal padding
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x):
        """
        x: (batch_size, in_channels, seq_len)
        """
        x = self.conv(x)
        return x[:, :, : -self.padding]  # Remove extra padding


class ResidualBlock(nn.Module):
    """Residual Block with Causal Convolutions"""

    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        """
        x: (batch_size, channels, seq_len)
        """
        res = x  # Residual connection
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm(x.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # Apply LayerNorm along channels
        return F.relu(x + res)  # Add residual connection


class TCNEncoder(nn.Module):
    """Temporal Convolutional Network"""

    def __init__(self, input_size, d_model, num_layers, kernel_size=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)  # Project input to d_model
        self.tcn_layers = nn.ModuleList(
            [
                ResidualBlock(d_model, kernel_size, dilation=2**i, dropout=dropout)
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, num_features)
        """
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)
        x = x.permute(0, 2, 1)  # Change to (batch_size, d_model, seq_len) for TCN

        for layer in self.tcn_layers:
            x = layer(x)  # Pass through residual TCN blocks

        x = x.permute(0, 2, 1)  # Convert back to (batch_size, seq_len, d_model)
        return x


class TCNDecoder(nn.Module):
    def __init__(self, d_model, num_layers, kernel_size=3, dropout=0.2):
        super().__init__()

        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.dropout = dropout

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = nn.Conv1d(
                in_channels=self.d_model if i == 0 else self.d_model,
                out_channels=self.d_model,
                kernel_size=self.kernel_size,
                padding=self.kernel_size - 1,  # Causal padding
                dilation=2**i,  # Increasing dilation
            )
            self.convs.append(conv)

        self.forecast_head = nn.Conv1d(self.d_model, self.d_model, kernel_size=1)
        self.output_layer = nn.Conv1d(self.d_model, 1, kernel_size=1)

    def forward(self, x):
        # Forward pass through each convolutional layer
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.forecast_head(x)  # [batch_size, d_model, seq_len]
        x = self.output_layer(x)  # [batch_size, 1, seq_len]

        return x.permute(0, 2, 1)  # [batch_size, seq_len, 1] (predicted time series)
