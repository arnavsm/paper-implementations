import torch
from torch import nn
from torch.nn import functional as F
from model_builder import build_transformer
from utils import count_parameters

# Example usage
src_vocab_size = 10000
tgt_vocab_size = 10000
src_seq_len = 128
tgt_seq_len = 128
d_model = 512
batch_size = 32

# Create the transformer model
transformer = build_transformer(src_vocab_size, 
                                tgt_vocab_size, 
                                src_seq_len, 
                                tgt_seq_len, 
                                d_model)

print(count_parameters(transformer))



