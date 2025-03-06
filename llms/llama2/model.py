from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TransformerConfig:
    model_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: Optional[int] = None
    vocab_size: int = -1  # Set later
    multiple_of: int = 256
    ffn_multiplier: Optional[float] = None
    norm_epsilon: float = 1e-5

    # KV Cache parameters
    max_batch_size: int = 32
    max_sequence_length: int = 2048

    device: str = None


class LayerNorm(nn.Module):
    def __init__(self, model_dim: int, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(model_dim))

    def forward(self, x: torch.Tensor):
        # x: (S, D)
        # Output: (S, D)
        return self.scale * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.epsilon)


def compute_rotary_frequencies(head_dim: int, seq_len: int, device: str, base: float = 10000.0):
    assert head_dim % 2 == 0, "Head dimension must be even."
    # theta: (D / 2,)
    # pos_ids: (S,)
    # frequencies: (S, D / 2)
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim)).to(device)
    pos_ids = torch.arange(seq_len, device=device)
    frequencies = torch.outer(pos_ids, theta).float()
    # frequencies: (S, D / 2)
    return torch.polar(torch.ones_like(frequencies), frequencies)


def apply_rotary_encoding(x: torch.Tensor, frequencies: torch.Tensor, device: str):
    # x: (S, H, D)
    # frequencies: (1, 1, S, D / 2)
    # x_complex: (S, H, D / 2, 2) (real and imaginary parts)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    frequencies = frequencies.unsqueeze(0).unsqueeze(2)
    # rotated: (S, H, D / 2, 2)
    rotated = x_complex * frequencies
    # x_out: (S, H, D)
    x_out = torch.view_as_real(rotated).reshape(*x.shape)
    # Output: (S, H, D)
    return x_out.type_as(x).to(device)


def expand_kv_heads(x: torch.Tensor, repeat_factor: int) -> torch.Tensor:
    # x: (S, H, D)
    if repeat_factor == 1:
        return x
    # Output: (S, H * repeat_factor, D)
    return x[:, :, None, :].expand(*x.shape[:2], repeat_factor, x.shape[-1]).reshape(x.shape[0], x.shape[1] * repeat_factor, x.shape[2])


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_kv_heads = config.num_heads if config.num_kv_heads is None else config.num_kv_heads
        self.num_query_heads = config.num_heads
        self.head_dim = config.model_dim // config.num_heads
        self.repeat_factor = self.num_query_heads // self.num_kv_heads

        self.query_proj = nn.Linear(config.model_dim, config.num_heads * self.head_dim, bias=False)
        self.key_proj = nn.Linear(config.model_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.value_proj = nn.Linear(config.model_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(config.num_heads * self.head_dim, config.model_dim, bias=False)

        self.cache_keys = torch.zeros((config.max_batch_size, config.max_sequence_length, self.num_kv_heads, self.head_dim))
        self.cache_values = torch.zeros((config.max_batch_size, config.max_sequence_length, self.num_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, frequencies: torch.Tensor):
        # x: (B, S, D)
        B, S, _ = x.shape

        queries = self.query_proj(x).view(B, S, self.num_query_heads, self.head_dim)
        # queries: (B, S, H_q, D_h)

        keys = self.key_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        # keys: (B, S, H_kv, D_h)

        values = self.value_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        # values: (B, S, H_kv, D_h)

        queries = apply_rotary_encoding(queries, frequencies, device=x.device)
        keys = apply_rotary_encoding(keys, frequencies, device=x.device)

        # Cache the keys and values
        self.cache_keys[:B, start_pos:start_pos + S] = keys
        self.cache_values[:B, start_pos:start_pos + S] = values

        # Expand keys and values across query heads
        keys = expand_kv_heads(self.cache_keys[:B, :start_pos + S], self.repeat_factor)
        # keys: (B, S, H_q, D_h)

        values = expand_kv_heads(self.cache_values[:B, :start_pos + S], self.repeat_factor)
        # values: (B, S, H_q, D_h)

        # Compute the attention scores
        scores = torch.matmul(queries.transpose(1, 2), keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # scores: (B, H_q, S, S)

        scores = F.softmax(scores.float(), dim=-1).type_as(queries)
        # scores: (B, H_q, S, S)

        # Compute the attention output
        output = torch.matmul(scores, values).transpose(1, 2).contiguous().view(B, S, -1)
        # output: (B, S, H_q * D_h)

        return self.output_proj(output)
        # output: (B, S, D)


class FeedForwardNetwork(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        hidden_dim = int(2 * (4 * config.model_dim) / 3)
        if config.ffn_multiplier:
            hidden_dim = int(config.ffn_multiplier * hidden_dim)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.fc1 = nn.Linear(config.model_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, config.model_dim, bias=False)
        self.fc3 = nn.Linear(config.model_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # x: (B, S, D)
        return self.fc2(F.silu(self.fc1(x)) * self.fc3(x))
        # output: (B, S, D)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForwardNetwork(config)
        self.norm1 = LayerNorm(config.model_dim, epsilon=config.norm_epsilon)
        self.norm2 = LayerNorm(config.model_dim, epsilon=config.norm_epsilon)

    def forward(self, x: torch.Tensor, start_pos: int, frequencies: torch.Tensor):
        # x: (B, S, D)
        x = x + self.attention(self.norm1(x), start_pos, frequencies)
        # x: (B, S, D)

        return x + self.feed_forward(self.norm2(x))
        # x: (B, S, D)


class TransformerModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.vocab_size != -1, "Vocab size must be specified."

        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.model_dim)
        # token_embeddings: (B, S, D)

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])

        self.norm = LayerNorm(config.model_dim, epsilon=config.norm_epsilon)
        self.output_layer = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.frequencies = compute_rotary_frequencies(config.model_dim // config.num_heads, config.max_sequence_length * 2, device=config.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # tokens: (B, S)
        B, S = tokens.shape
        assert S == 1, "Only one token at a time is allowed."

        embeddings = self.token_embeddings(tokens)
        # embeddings: (B, S, D)

        freqs = self.frequencies[start_pos:start_pos + S]
        # freqs: (S, H)

        for layer in self.layers:
            embeddings = layer(embeddings, start_pos, freqs)
        # embeddings: (B, S, D)

        return self.output_layer(self.norm(embeddings)).float()
        # output: (B, S, V)
        # where V is vocab_size