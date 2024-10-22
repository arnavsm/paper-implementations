import re, collections


def read_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def initial_setup(text: str):
    text = text + "_"
    return (sorted(list(set(text))), text)


class BPETokenizer:
    def __init__(self, num_merges=100):
        self.num_merges = num_merges
        self.vocab = []
        self.merges = {}

    def __get_stats(self, corpus: str):
        pairs = collections.defaultdict(int)
        words = corpus.split()

        for word in words:
            symbols = list(word)
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += 1

        return pairs

    def __merge_most_frequent_pair(self, corpus: str, pair: tuple):
        pair_str = ' '.join(pair)
        merged_pair = ''.join(pair)
        parts = corpus.split()

        for i, word in enumerate(parts):
            parts[i] = word.replace(pair_str, merged_pair)

        return ' '.join(parts)

    def __bpe(self, char_list: list, corpus: str):
        words = corpus.split()
        corpus = ' '.join(' '.join(word) for word in words)
        
        for i in range(self.num_merges):
            pair_freqs = self.__get_stats(corpus)
            if not pair_freqs:
                break
            most_frequent = max(pair_freqs, key=pair_freqs.get)
            self.merges[most_frequent] = ''.join(most_frequent)
            corpus = self.__merge_most_frequent_pair(corpus, most_frequent)
            char_list.append(''.join(most_frequent))

        return char_list, corpus

    def __learner(self, corpus: str):
        initial_vocab = sorted(list(set(corpus)))
        self.vocab, self.corpus = self.__bpe(initial_vocab, corpus)

    def __segmenter(self, word: str):
        word = ' '.join(list(word))

        while True:
            pairs = [(i, i+1, word[i:i+2]) for i in range(0, len(word)-1, 2) if word[i:i+2] in self.merges.values()]
            if not pairs:
                break
            for start, end, pair in pairs:
                if pair in self.merges.values():
                    word = word[:start] + ''.join(pair.split()) + word[end:]

        return word.split()

    def build_vocab(self, corpus: str):
        self.__learner(corpus)

    def tokenize(self, text: str):
        tokens = []

        for word in text.split():
            tokens.extend(self.__segmenter(word))

        return tokens

def main():
    filepath = "bpe/corpus.txt"
    iters = 200

    text = read_file(filepath=filepath)
    _, corpus = initial_setup(text)

    tokenizer = BPETokenizer(num_merges=iters)
    tokenizer.build_vocab(corpus)

    test_text = "Arnav says Hi!"
    tokens = tokenizer.tokenize(test_text)
    print("Tokens:", tokens)

if __name__ == "__main__":
    main()