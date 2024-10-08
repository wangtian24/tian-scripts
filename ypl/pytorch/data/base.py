from collections.abc import Callable
from typing import Any, Generic, Self, TypeVar

import pandas as pd
import torch.utils.data as tud
from transformers import AutoTokenizer

from ypl.pytorch.torch_utils import DeviceMixin, StrTensorDict
from ypl.utils import RNGMixin

ExampleType = TypeVar("ExampleType")


class DatasetCollator(Generic[ExampleType]):
    """
    Abstract base class for dataset collators. Provides a callable interface to collate a batch of examples into
    a formatted structure suitable for model input. For simplicity, the output of this collator is assumed to be an
    arbitrary dictionary with string keys and any values.

    See Also:
        - `TokenizerCollator` for a concrete implementation that tokenizes input examples.
        - PyTorch's `DataLoader` for a how collators are used.
    """

    def __call__(self, batch: list[ExampleType]) -> StrTensorDict:
        """
        Collate a list of examples into a batch.

        Args:
            batch: A list of examples to be collated.

        Returns:
            A dictionary containing the collated batch data.
        """
        raise NotImplementedError


class TokenizerCollator(DeviceMixin, DatasetCollator[ExampleType]):
    """
    Dataset collator that tokenizes input examples using a provided tokenizer. Inherits from DeviceMixin to manage
    device placement and DatasetCollator for batch collation interface.
    """

    def __init__(self, tokenizer: AutoTokenizer) -> None:
        """
        Initialize the TokenizerCollator with a tokenizer.

        Args:
            tokenizer: The tokenizer to be used for encoding input examples.
        """
        self.tokenizer = tokenizer

    def __call__(self, batch: list[ExampleType]) -> StrTensorDict:
        """
        Tokenize and collate a batch of examples.

        Args:
            batch: A list of examples to be tokenized and collated.

        Returns:
            A dictionary containing the tokenized and collated batch data.
        """
        batch_ = self.collate(batch)
        return self.to_alike(batch_)

    def collate(self, batch: list[ExampleType]) -> StrTensorDict:
        """
        Collate a list of examples without tokenization. This method should be overridden by subclasses to implement
        specific collation logic.

        Args:
            batch: A list of examples to be collated.

        Returns:
            A dictionary containing the collated batch data.
        """
        raise NotImplementedError


class TypedDataset(tud.Dataset, Generic[ExampleType]):
    """
    Abstract base class for typed datasets. Provides foundational methods for retrieving and manipulating
    dataset examples.
    """

    def __getitem__(self, index: int) -> ExampleType:
        """Retrieve an example by its index."""
        raise NotImplementedError

    def filter(self, predicate: Callable[[ExampleType], bool]) -> Self:
        """Filter the dataset based on a given predicate."""
        raise NotImplementedError

    def map(self, func: Callable[[ExampleType], ExampleType]) -> Self:
        """Apply a transformation function to all examples in the dataset."""
        raise NotImplementedError


class ListDataset(TypedDataset[ExampleType]):
    """Dataset implementation that stores examples in a list."""

    def __init__(self, examples: list[ExampleType]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        return len(self.examples)

    def __getitem__(self, index: int) -> ExampleType:
        """Retrieve an example by its index."""
        return self.examples[index]

    def filter(self, predicate: Callable[[ExampleType], bool]) -> "ListDataset[ExampleType]":
        """Filter the dataset based on a given predicate."""
        return ListDataset(list(filter(predicate, self.examples)))

    def map(self, func: Callable[[ExampleType], ExampleType]) -> "ListDataset[ExampleType]":
        """Apply a transformation function to all examples in the dataset."""
        return ListDataset(list(map(func, self.examples)))


class PandasDataset(RNGMixin, TypedDataset[ExampleType]):
    """
    Dataset implementation that utilizes a pandas DataFrame for storage. Inherits from RNGMixin to provide random
    number generation capabilities and TypedDataset for dataset interface.
    """

    label_column: str | None = None

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        return len(self.df)

    def create_label_map(self) -> dict[str, int]:
        """
        Creates a mapping from category names to unique integer identifiers.

        Returns:
            A dictionary mapping labels to unique integer IDs.
        """
        if self.label_column is None:
            raise ValueError("Label column not set")

        all_categories = self.df[self.label_column].unique().tolist()
        category_map = {category: i for i, category in enumerate(all_categories)}

        return category_map

    def split(self, percentage: int, func: Callable[[ExampleType], int] | None = None) -> tuple[Self, Self]:
        """
        Split the dataset into training and validation sets based on a percentage.

        Args:
            percentage: The percentage of data to include in the training set.
            func: A function to determine the split. Defaults to a random function.

        Returns: A tuple containing the training and validation datasets.
        """
        if func is None:
            func = lambda _: self.get_rng().randint(0, 99)  # noqa: E731

        func_bounded = lambda row: func(row) % 100  # noqa: E731
        df = self.df.copy()
        df["split"] = df.apply(func_bounded, axis=1)
        train = df[df["split"] < percentage]
        val = df[df["split"] >= percentage]

        train.drop(columns=["split"], inplace=True)
        val.drop(columns=["split"], inplace=True)

        return (
            self.__class__(train),
            self.__class__(val),
        )

    @classmethod
    def from_csv(cls, path: str, **kwargs: Any) -> Self:
        """
        Create a PandasDataset instance from a CSV file.

        Args:
            path: The path to the CSV file.
            **kwargs: Additional keyword arguments to pass to `pd.read_csv`.

        Returns:
            A new PandasDataset instance containing data from the CSV file.
        """
        return cls(pd.read_csv(path, **kwargs))
