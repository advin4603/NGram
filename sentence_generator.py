from ngram import NGram
from pathlib import Path
from random import choices

from typing import *


class SentenceGenerator(NGram[str]):
    def __init__(self, n: int, tokenizer: Callable[[str], Iterable[str]]):
        super(SentenceGenerator, self).__init__(n)
        self.tokenizer = lambda sentence: tuple(tokenizer(sentence))

    def train(self, sentence: str):
        super(SentenceGenerator, self).train(self.tokenizer(sentence))

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
        return super(SentenceGenerator, self).predict(self.tokenizer(prev_str))

    def generate(self) -> str:
        sentence = ""
        while predictions := self.predict(sentence):
            sentence += choices(predictions, weights=range(len(predictions), 0, -1), k=1)[0] + " "
        return sentence


if __name__ == "__main__":
    import string

    char_set = set(string.ascii_letters)
    gen = SentenceGenerator(3,
                            lambda sentence: ("".join(char.lower() for char in word if char in char_set) for word
                                              in
                                              sentence.split()))
    print("Training...")
    gen.train_from_file("steam_descriptions.txt")
    print("Done!")
    print(gen.generate())
