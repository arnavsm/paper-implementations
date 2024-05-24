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

        # creating the positional encoding matrix
        pos_encoding = torch.zeros(seq_length, d_model)

        # vector to represent position in sequence
        position = torch.arange(0, seq_length, dtype= torch.float32).unsqueeze(1) # Seq_Length * 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # sine & cosine
        pos_encoding[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        pos_encoding[:, 1::2] = torch.cos(position * div_term) # Similarly for cosine

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # register the positional encoding as a buffer
        self.register_buffer('pe', pe) # will be saved as in a file

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
    
# Layer Normalization
class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * ((x - mean) / (std + self.eps)) + self.bias

# Feedforward Network
class FeedforwardNetwork(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float) -> None:
        super().__init__()
        self.Linear_1 = nn.Linear(d_model, d_ff) # W1 & B1
        self.Dropout = nn.Dropout(dropout_rate)
        self.Linear_2 = nn.Linear(d_ff, d_model) # W2 & B2
    
    def forward(self, x):
        x = self.Dropout(torch.relu(self.Linear_1(x)))
        x = self.Linear_2
        return x

# Residual Connections
class ResidualConnection(nn.Module):
    
    def __init__(self, features: int, dropout_rate: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = LayerNormalization(features)
    
    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))

# Multiheaded Attention
class MultiHeadedAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout_rate: float) -> None:
        super().__init__()
        self.d_model = d_model
        assert d_model % h == 0, "d_model is not divisible by h"

        self.h = h
        
        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias = False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias = False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias = False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias = False) # Wo
        self.dropout = nn.Dropout(dropout_rate)

    @staticmethod # reusability
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        attention_scores = (query @ key.transpore(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) # write a very low value (indicating -inf) to the positions where mask == 0

        attention_scores = attention_scores.softmax(dim = -1) # (batch, h, seq_len, seq_len) # apply softmax

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores # return attention scores which can be used for visualization


        
    def forward(self, q, k, v, mask):
        query =  self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadedAttentionBlock.attention(query, key, value, mask, self.dropout)

        # combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)
    
# Encoder Block
class EncoderBlock(nn.Module):

    def __init__(self, dropout_rate: float,
                    self_attention_block = MultiHeadedAttentionBlock, 
                    feed_forward_block = FeedforwardNetwork) -> None:
        
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout_rate = dropout_rate
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout_rate) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

# Encoder Architecture
class Encoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.normalization = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.normalization(x)
    
# Decoder Block
class DecoderBlock(nn.Module):

    def __init__(self, features: int, 
                 self_attention_block: MultiHeadedAttentionBlock, 
                 cross_attention_block: MultiHeadedAttentionBlock, 
                 feed_forward_block: FeedforwardNetwork, 
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, 
                 decoder: Decoder, 
                 src_embed: InputEmbedding, 
                 tgt_embed: InputEmbedding, 
                 src_pos: PositionalEncoding, 
                 tgt_pos: PositionalEncoding, 
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, 
                      tgt_vocab_size: int, 
                      src_seq_len: int, 
                      tgt_seq_len: int, 
                      d_model: int=512, 
                      N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadedAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedforwardNetwork(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadedAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadedAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedforwardNetwork(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

    
