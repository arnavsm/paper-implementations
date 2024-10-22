import re
from collections import defaultdict
from typing import List, Tuple, Dict

def read_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().lower()

class BPETokenizer:
    def __init__(self, num_merges: int = 100):
        self.num_merges = num_merges
        self.vocab: Dict[str, int] = {}
        self.merges: Dict[Tuple[str, str], str] = {}
        self.special_tokens = {"<unk>": 0, "<s>": 1, "</s>": 2, "</w>": 3}
        
    def _preprocess_text(self, text: str) -> str:
        # text = text.lower()
        text = re.sub(r'([.,!?()])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _get_stats(self, words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        pairs = defaultdict(int)
        for word in words:
            for i in range(len(word) - 1):
                pairs[tuple(word[i:i+2])] += 1
        return pairs

    def _merge_pair(self, words: List[List[str]], pair: Tuple[str, str], merged: str) -> List[List[str]]:
        new_words = []
        for word in words:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and tuple(word[i:i+2]) == pair:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        return new_words

    def build_vocab(self, corpus: str) -> None:
        corpus = self._preprocess_text(corpus)
        words = [list(word) + ['</w>'] for word in corpus.split()]
        chars = set(char for word in words for char in word)
        self.vocab = {char: idx + len(self.special_tokens) for idx, char in enumerate(sorted(chars))}
        self.vocab.update(self.special_tokens)
        
        for i in range(self.num_merges):
            pairs = self._get_stats(words)
            if not pairs:
                break
                
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            merged = ''.join(best_pair)
            self.merges[best_pair] = merged
            words = self._merge_pair(words, best_pair, merged)
            
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)

    def tokenize(self, text: str) -> List[str]:
        if not self.merges:
            raise ValueError("Tokenizer needs to be trained first using build_vocab")
            
        text = self._preprocess_text(text)
        final_tokens = []
        
        for word in text.split():
            word = list(word) + ['</w>']
            while True:
                pairs = self._get_stats([word])
                if not pairs:
                    break
                
                valid_pairs = {pair: freq for pair, freq in pairs.items() if pair in self.merges}
                if not valid_pairs:
                    break
                
                best_pair = max(valid_pairs.items(), key=lambda x: x[1])[0]
                word = self._merge_pair([word], best_pair, self.merges[best_pair])[0]
            
            final_tokens.extend(word)
        
        return final_tokens

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        inv_vocab = {v: k for k, v in self.vocab.items()}
        tokens = [inv_vocab.get(id, '<unk>') for id in token_ids]
        text = ' '.join(''.join(tokens).replace('</w>', ' ').split())
        return text.strip()

def main():
    filepath = "bpe/corpus.txt"
    text = read_file(filepath)
    
    tokenizer = BPETokenizer(num_merges=100)
    tokenizer.build_vocab(text)
    
    test_text = "Arnav says Hi!"
    tokens = tokenizer.tokenize(test_text)
    print("Tokens:", tokens)
    
    ids = tokenizer.encode(test_text)
    print("Token IDs:", ids)
    
    decoded = tokenizer.decode(ids)
    print("Decoded:", decoded)

if __name__ == "__main__":
    main()