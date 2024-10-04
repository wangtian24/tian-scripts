from typing import Self, TypeVar

import numpy as np


class RNGMixin:
    """
    Mixin class to add a random number generator to a class.
    """

    _rng: np.random.RandomState | None = None
    _seed: int | None = None

    def set_seed(self, seed: int) -> None:
        if self._seed is not None:
            raise ValueError("Seed already set")

        self._seed = seed

    def with_seed(self, seed: int) -> Self:
        self.set_seed(seed)
        return self

    def get_rng(self) -> np.random.RandomState:
        if self._rng is None:
            self._rng = np.random.RandomState(self._seed)

        return self._rng


K = TypeVar("K")
V = TypeVar("V")


def dict_extract(d: dict[K, V], keys: set[K]) -> dict[K, V]:
    """Extracts a dictionary containing only the keys in `keys` from the dictionary `d`."""
    return {k: v for k, v in d.items() if k in keys}
