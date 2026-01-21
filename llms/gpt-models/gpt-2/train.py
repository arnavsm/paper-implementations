import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dataclasses import dataclass


@dataclass
class GPTConfig:
    n_head: int = 6
    n_layer: int = 12
    n_emb: int = 384

    context_max: int = 256
    vocab_size: int = 50e4

config = GPTConfig()


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_emb % config.n_head == 0
        self.w_qkv = nn.Linear(config.n_emb, 3 * config.n_emb)
        self.w_o = nn.Linear(config.n_emb, config.n_emb)
        
        self.n_head = config.n_head
        self.d_head = config.n_emb % config.n_head
        self.n_emb = config.n_emb
        self.register_buffer("casual_mask", torch.tril(torch.ones(config.context_max, config.context_max))
                                     .view(1, 1, config.context_max, config.context_max))
        
    def forward(self, x):
        B, N, D = x.size()

        qkv = self.w_qkv(x)
        q, k, v = qkv.split(self.n_emb, dim = 2)
        # B, T, N
        q = q.view(B, N, self.n_head, D // self.n_head).transpose(1, 2) 
        k = k.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        v = v.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        # B, H, N, D_H
        att = (q @ v.transpose(-2, -1) ) * (1 / math.sqrt(self.d_head)) 
        # B, H, N, N
        att = att.masked_fill(self.bias[:, :, :N, :N] == 0, float('-inf'))
        # B, H, N, N
        att = F.softmax(att, dim=-1)
        y = att @ v 
        # B, H, N, D_H
        y = y.transpose(1, 2).reshape(B, N, D)
        y = self.w_o(y)
        return y



class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_1 = nn.Linear(config.n_emb, 4*config.n_emb, bias=True)
        self.layer_2 = nn.Linear(4*config.n_emb, config.n_emb, bias=True)
        self.gelu = nn.GELU(approximate='tanh')
        
    def forward(self, x):
        return self.layer_2(self.gelu(self.layer_1(x)))


class Block(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_emb)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_emb)
        self.ffn = FFN(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_emb),
            wpe = nn.Embedding(config.context_max, config.n_emb),
            h = nn.ModuleList[[Block(config) for _ in range(config.n_layer)]],
            ln_f = nn.LayerNorm(config.n_emb),
        ))
        self.lm_head = nn.Linear(config.emb, config.vocab_size, bias=False)

    def forward(self, idx):
        B, N = idx.size()
        assert N < self.config.context_max
        pos = torch.arange(0, N, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
    





