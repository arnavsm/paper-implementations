"""
Contains PyTorch model code to instantiate a model.
"""
import torch
from torch import nn
import math

# we won't use einops and rely on view and permute for rearrangement of tensors

# Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, 
                 P: int = 16, 
                 C: int = 3,
                 d_model: int = 512) -> None:
        super().__init__()
        self.channels = C
        self.patch_size = P
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.linear_proj = nn.Linear(self.channels * (self.patch_size**2), self.d_model)
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Height and Width must be divisible by patch_size"
        # N = (H // self.patch_size) * (W // self.patch_size)
        
        x = x.view(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)  # (B, C, H // self.patch_size, P, W // self.patch_size, P)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, H // self.patch_size, W // self.patch_size, C, P, P)
        x = x.view(B, (H // self.patch_size) * (W // self.patch_size), C * self.patch_size * self.patch_size)  # (B, N, C*P*P)

        x = self.linear_proj(x)  # (B, N, d_model)

        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat((cls_token, x), dim=1)
        
        return x

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, N: int) -> None: 
        super().__init__()
        self.N = N  # N = (H // self.patch_size) * (W // self.patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, N + 1, d_model))

    def forward(self, x):
        x = x + self.pos_embedding[:, :x.shape[1], :]
        return x

# Layer Normalization
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

# Feedforward Network
class FeedforwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# Residual Connections
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout_rate: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = LayerNormalization(features)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# Multiheaded Attention
class MultiHeadedAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout_rate: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    @staticmethod
    def attention(query, key, value, dropout=None):
        d_k = query.size(-1)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        attention_scores = scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return torch.matmul(attention_scores, value), attention_scores
    
    def forward(self, q, k, v):
        B, T, _ = q.size()

        query = self.w_q(q).view(B, T, self.h, self.d_k).transpose(1, 2)  # (B, h, T, d_k)
        key = self.w_k(k).view(B, T, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(B, T, self.h, self.d_k).transpose(1, 2)

        x, _ = self.attention(query, key, value, self.dropout)
        x = x.transpose(1, 2).contiguous().view(B, T, self.h * self.d_k)

        return self.w_o(x)

# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadedAttentionBlock, feed_forward_block: FeedforwardNetwork, dropout_rate: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout_rate) for _ in range(2)])

    def forward(self, x):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

# Encoder Architecture
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# MLP Head
class MLPHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, d_ff: int, dropout_rate: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(d_ff, num_classes)
    
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# Full Vision Transformer (ViT) Model
class VisionTransformer(nn.Module):
    def __init__(self, image_size: int, patch_size: int, num_classes: int, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout_rate: float) -> None:
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.patch_embed = PatchEmbedding(patch_size, 3, d_model)
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.pos_encoding = PositionalEncoding(d_model, num_patches)
        self.encoder = Encoder(d_model, nn.ModuleList([
            EncoderBlock(d_model, MultiHeadedAttentionBlock(d_model, n_heads, dropout_rate), 
                         FeedforwardNetwork(d_model, d_ff, dropout_rate), dropout_rate) 
            for _ in range(n_layers)
        ]))
        self.mlp_head = MLPHead(d_model, num_classes, d_ff, dropout_rate)
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        cls_token = x[:, 0]  # Extract the class token
        return self.mlp_head(cls_token)

# Function to create and test the Vision Transformer model
def create_and_test_vit(image_size: int, patch_size: int, num_classes: int, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout_rate: float) -> torch.Tensor:
    model = VisionTransformer(image_size, patch_size, num_classes, d_model, n_heads, n_layers, d_ff, dropout_rate)
    dummy_input = torch.randn(8, 3, image_size, image_size)  # Batch size 8, 3 color channels, image_size x image_size
    output = model(dummy_input)
    return output

# Example usage
image_size = 128
patch_size = 16
num_classes = 10
d_model = 512
n_heads = 8
n_layers = 6
d_ff = 2048
dropout_rate = 0.1

output = create_and_test_vit(image_size, patch_size, num_classes, d_model, n_heads, n_layers, d_ff, dropout_rate)
print(output.shape)  # Should print (8, num_classes)
