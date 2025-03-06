import torch
from torch import nn
import math

class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, dropout_rate):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = word_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BERTSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_rate):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout_rate)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer

class BERTLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout_rate):
        super().__init__()
        self.attention = BERTSelfAttention(hidden_size, num_attention_heads, dropout_rate)
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm1 = nn.LayerNorm(hidden_size)
        self.LayerNorm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.LayerNorm1(hidden_states + self.dropout(attention_output))
        
        intermediate_output = torch.relu(self.intermediate(attention_output))
        layer_output = self.output(intermediate_output)
        layer_output = self.LayerNorm2(attention_output + self.dropout(layer_output))
        
        return layer_output

class BERTModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads, 
                 intermediate_size, max_position_embeddings, dropout_rate):
        super().__init__()
        self.embeddings = BERTEmbeddings(vocab_size, hidden_size, max_position_embeddings, dropout_rate)
        self.encoder = nn.ModuleList([BERTLayer(hidden_size, num_attention_heads, intermediate_size, dropout_rate) 
                                      for _ in range(num_hidden_layers)])

    def forward(self, input_ids, attention_mask):
        embedding_output = self.embeddings(input_ids)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = embedding_output
        for layer in self.encoder:
            hidden_states = layer(hidden_states, extended_attention_mask)

        return hidden_states