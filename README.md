# NGram

Implementation of an [n-gram model](https://en.wikipedia.org/wiki/N-gram) in pure python using only the standard library. The n-gram model has been used to implement a word predictor and a sentence generator which can be trained on the provided steam descriptions corpus.

## Prerequisites
- python 3.10+

## Running
- Run the sentence generator
```bash
python sentence_generator.py
```

- Run the word predictor
```bash
python word_predictor.py
```

- To train using your own corpus, edit the call to `gen.train_from_file` or `predictor.train_from_file`
