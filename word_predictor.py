from ngram import NGram
from pathlib import Path

from typing import *


class WordPredictor(NGram[str]):
    def __init__(self, n: int, tokenizer: Callable[[str], Iterable[str]]):
        super(WordPredictor, self).__init__(n)
        self.tokenizer = lambda sentence: tuple(tokenizer(sentence))

    def train(self, sentence: str):
        super(WordPredictor, self).train(self.tokenizer(sentence))

    def train_from_file(self, filepath: str,
                        line_by_line: bool = True,
                        sentence_tokenizer: Callable[[str], Iterable[str]] = lambda sentence: (sentence,)):
        with open(Path(filepath)) as file:
            if line_by_line:
                for line in file:
                    for sentence in sentence_tokenizer(line):
                        self.train(sentence)
            else:
                for sentence in sentence_tokenizer(file.read()):
                    self.train(sentence)

    def predict(self, prev_str: str) -> list[str]:
        return super(WordPredictor, self).predict(self.tokenizer(prev_str))


if __name__ == "__main__":
    import string

    char_set = set(string.ascii_letters)
    predictor = WordPredictor(3,
                              lambda sentence: ("".join(char.lower() for char in word if char in char_set) for word in
                                                sentence.split()))
    print("Training...")
    predictor.train_from_file("../corpus.txt")
    print("Done!")
    while True:
        prompt = input(">")
        print(predictor.predict(prompt))
