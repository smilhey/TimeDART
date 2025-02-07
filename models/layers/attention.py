import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product(d_k, Q, K, V, mask=None):
    """
    Q: (batch_size, num_heads, seq_len, d_k)
    K: (batch_size, num_heads, seq_len, d_k)
    V: (batch_size, num_heads, seq_len, d_k)
    mask: (batch_size, 1, 1, seq_len) or None

    Returns:
    output: (batch_size, num_heads, seq_len, d_k)
    """
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=torch.float32)
    )  # (batch_size, num_heads, seq_len, seq_len)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attention = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
    output = torch.matmul(attention, V)  # (batch_size, num_heads, seq_len, d_k)

    return output


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, d_head):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, d_head)
        self.W_k = nn.Linear(embed_dim, d_head)
        self.W_v = nn.Linear(embed_dim, d_head)
        self.d_head = d_head

    def forward(self, queries, keys, values, mask=None):
        """
        queries: (batch_size, seq_len, embed_dim)
        keys: (batch_size, seq_len, embed_dim)
        values: (batch_size, seq_len, embed_dim)

        Returns:
        output: (batch_size, seq_len, d_head)
        """
        Q = self.W_q(queries)  # (batch_size, seq_len, d_head)
        K = self.W_k(keys)  # (batch_size, seq_len, d_head)
        V = self.W_v(values)  # (batch_size, seq_len, d_head)

        output = scaled_dot_product(
            self.d_head, Q, K, V, mask
        )  # (batch_size, seq_len, d_head)
        return output


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "MultiHeadAttention : embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, queries, keys, values, mask=None):
        """
        queries: (batch_size, seq_len, embed_dim)
        keys: (batch_size, seq_len, embed_dim)
        values: (batch_size, seq_len, embed_dim)
        mask: (batch_size, 1, 1, seq_len) or None

        Returns:
        output: (batch_size, seq_len, embed_dim)
        """
        batch_size = queries.size(0)
        Q = (
            self.W_q(queries)  # (batch_size, seq_len, embed_dim)
            .view(
                batch_size, -1, self.num_heads, self.d_head
            )  # (batch_size, seq_len, num_heads, d_head)
            .transpose(1, 2)  # (batch_size, num_heads, seq_len, d_head)
        )
        K = (
            self.W_k(keys)  # (batch_size, seq_len, embed_dim)
            .view(
                batch_size, -1, self.num_heads, self.d_head
            )  # (batch_size, seq_len, num_heads, d_head)
            .transpose(1, 2)  # (batch_size, num_heads, seq_len, d_head)
        )
        V = (
            self.W_v(values)  # (batch_size, seq_len, embed_dim)
            .view(
                batch_size, -1, self.num_heads, self.d_head
            )  # (batch_size, seq_len, num_heads, d_head)
            .transpose(1, 2)  # (batch_size, num_heads, seq_len, d_head)
        )
        context = scaled_dot_product(
            self.d_head, Q, K, V, mask
        )  # (batch_size, num_heads, seq_len, d_head)
        context = (
            context.transpose(1, 2)  # (batch_size, seq_len, num_heads, d_head)
            .contiguous()
            .view(batch_size, -1, self.embed_dim)  # (batch_size, seq_len, embed_dim)
        )
        output = self.W_o(context)  # (batch_size, seq_len, embed_dim)
        return output  # (batch_size, seq_len, embed_dim)
