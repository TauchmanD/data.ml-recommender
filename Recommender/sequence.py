from abc import ABC, abstractmethod
import pandas as pd

class Sequence(ABC):
    def __init__(self, preferences: pd.DataFrame):
        self.preferences = preferences
        self.recommended_items = {} # cache of items that were already recommended
    @abstractmethod
    def __next__(self) -> pd.DataFrame:
        pass



class CustomSequence(Sequence):
    def __next__(self) -> pd.DataFrame:
        return pd.DataFrame([[5]])


def get_sequence(sequence_name, preferences):
    if sequence_name == "custom":
        return CustomSequence(preferences)

if __name__ == "__main__":
    preferences = pd.DataFrame(
        [
            [1, 1, 0.8],
            [1, 2, 0.1],
            [1, 3, 0.4],
            [1, 4, 0.9],
            [2, 1, 0.3],
            [2, 2, 0.9],
            [2, 3, 0.1],
            [2, 4, 0.1],
            [3, 1, 0.8],
            [3, 2, 0.4],
            [3, 3, 0.4],
            [3, 4, 0.1],
            [4, 1, 0.3],
            [4, 2, 0.5],
            [4, 3, 1.0],
            [4, 4, 0.1],
        ],
        columns=["userId", "movieId", "rating"],
    )
    preferences = preferences.pivot_table(
        index="userId", columns="movieId", values="rating"
    )

    seq = get_sequence("custom", preferences)
    print(next(seq))
