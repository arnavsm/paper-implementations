import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from bert_model import BERTModel  

class CorpusDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        encoding = self.tokenizer(line, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        return {key: val.squeeze(0) for key, val in encoding.items()}

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # we only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # the rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

 # hyperparameters
vocab_size = 30522  # bert's default vocab size
hidden_size = 768
num_hidden_layers = 12
num_attention_heads = 12
intermediate_size = 3072
max_position_embeddings = 512
dropout_rate = 0.1
batch_size = 32
num_epochs = 3
learning_rate = 1e-4

def main():
   
    # initialize tokenizer & model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BERTModel(vocab_size, hidden_size, num_hidden_layers, num_attention_heads, 
                      intermediate_size, max_position_embeddings, dropout_rate)

    # preparing data
    dataset = CorpusDataset('path/to/your/corpus.txt', tokenizer, max_position_embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            # mask tokens for mlm
            inputs, labels = mask_tokens(input_ids, tokenizer)

            # forward pass
            outputs = model(inputs, attention_mask)

            # compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(outputs.view(-1, vocab_size), labels.view(-1))

            # backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"epoch: {epoch+1}, loss: {loss.item()}")

    torch.save(model.state_dict(), 'bert_mlm_pretrained.pth')

if __name__ == "__main__":
    main()