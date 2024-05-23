import torch
from torch import nn
import math

# Embedding Layer
class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super(InputEmbedding).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding * math.sqrt(self.d_model)
    
# Positional Encoding
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_length: int, dropout_rate: float) -> None:
        super(PositionalEncoding).__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

        # Creating the positional encoding matrix
        pos_encoding = torch.zeros(seq_length, d_model)

        # Vector to represent position in sequence
        position = torch.arange(0, seq_length, dtype= torch.float32).unsqueeze(1) # Seq_Length * 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Sine & Cosine
        pos_encoding[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pos_encoding[:, 1::2] = torch.cos(position * div_term) # Similarly for cosine

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
    
# Residual Connections
#class ResidualConnections(nn.Module):

    #def __init__(self):

    
