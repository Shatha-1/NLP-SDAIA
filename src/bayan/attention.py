"""Lab 2 starter: scaled dot-product attention and multi-head attention."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, mask=None, return_weights=False):
    """Scaled dot-product attention.

    q, k, v: (..., seq_len_q/k, d_k)
    mask: broadcastable to (..., seq_len_q, seq_len_k); positions where mask == 0
          are prevented from attending (set to -inf before softmax).
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    output = weights @ v

    if return_weights:
        return output, weights
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, heads, seq_len, d_k)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.d_model)

    def forward(self, x, mask=None, return_weights=False):
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        result = attention(q, k, v, mask=mask, return_weights=return_weights)
        if return_weights:
            out, weights = result
        else:
            out, weights = result, None

        out = self.out_proj(self._merge_heads(out))

        if return_weights:
            return out, weights
        return out
