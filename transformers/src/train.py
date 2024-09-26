import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
import time

from model_builder import build_transformer  # Import your transformer model

# Constants
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 0.0001
SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'fr'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tokenizers
token_transform = {}
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='fr_core_news_sm')

# Helper function to yield tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

# Build vocabularies
vocab_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=['<unk>', '<pad>', '<bos>', '<eos>'],
                                                    special_first=True)
    vocab_transform[ln].set_default_index(vocab_transform[ln]['<unk>'])

# Helper function to combine tensors into a batch
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

# Indices for special tokens
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

# Transformer Dataset
class TranslationDataset(Dataset):
    def __init__(self, filename):
        self.data = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                en, fr = line.strip().split('\t')
                self.data.append((en, fr))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        en, fr = self.data[idx]
        src_tokens = token_transform[SRC_LANGUAGE](en)
        tgt_tokens = token_transform[TGT_LANGUAGE](fr)
        src_ids = [vocab_transform[SRC_LANGUAGE][token] for token in src_tokens]
        tgt_ids = [vocab_transform[TGT_LANGUAGE][token] for token in tgt_tokens]
        return (torch.tensor(src_ids), torch.tensor(tgt_ids))

# Create datasets and dataloaders
train_dataset = TranslationDataset('engfre.txt')
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Initialize the Transformer model
src_vocab_size = len(vocab_transform[SRC_LANGUAGE])
tgt_vocab_size = len(vocab_transform[TGT_LANGUAGE])
transformer = build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len=128, tgt_seq_len=128)
transformer = transformer.to(DEVICE)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(transformer.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

# Training function
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        optimizer.zero_grad()
        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            tgt_out = tgt[1:, :]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Helper function to generate mask
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# Training loop
for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    train_loss = train_epoch(transformer, train_dataloader, optimizer, criterion)
    end_time = time.time()
    val_loss = evaluate(transformer, train_dataloader, criterion)  # Using train_dataloader for simplicity
    print(f"Epoch: {epoch+1}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
          f"Epoch time = {(end_time - start_time):.3f}s")

# Save the model
torch.save(transformer.state_dict(), 'transformer_model.pth')

print("Training completed!")