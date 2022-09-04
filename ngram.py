from typing import *

T = TypeVar('T')


class NGram(Generic[T]):
    def __init__(self, n: int) -> None:
        self.N = n
        self.frequencies: list[dict[tuple[T, ...], dict[T, int]]] = [dict() for _ in range(self.N)]

    def train(self, sequence: tuple[T, ...]):
        for index in range(min(self.N - 1, len(sequence))):
            freq_dict = self.frequencies[index]
            prev_states = sequence[:index]
            if prev_states not in freq_dict:
                freq_dict[prev_states] = {sequence[index]: 1}
            else:
                freq_dict[prev_states].setdefault(sequence[index], 0)
                freq_dict[prev_states][sequence[index]] += 1

        freq_dict = self.frequencies[self.N - 1]
        for index, state in enumerate(sequence[self.N - 1:], self.N - 1):
            prev_states = tuple(sequence[index - self.N + 1:index])
            if prev_states not in freq_dict:
                freq_dict[prev_states] = {state: 1}
            else:
                freq_dict[prev_states].setdefault(state, 0)
                freq_dict[prev_states][state] += 1

    def predict(self, prev_states: tuple[T, ...]) -> list[T]:
        prev_states = prev_states[len(prev_states) - self.N + 1:]
        prob_dict = self.frequencies[len(prev_states)]
        if prev_states in prob_dict:
            predictions = list(prob_dict[prev_states].keys())
            predictions.sort(reverse=True, key=lambda word: prob_dict[prev_states][word])
        else:
            predictions = []

        return predictions
