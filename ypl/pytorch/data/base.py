from collections.abc import Callable
from typing import Any, Generic, Literal, Self, TypeVar

import datasets
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

    @classmethod
    def create_default(cls) -> Self:
        """Create a default dataset instance."""
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
    default_invocation: Literal["parquet", "csv", "hf", "init", ""] = ""
    default_invocation_args: tuple[Any, ...] = ()
    default_invocation_kwargs: dict[str, Any] = {}
    default_split_name: str = "train"

    def __init__(self, df: pd.DataFrame, post_init: bool = True) -> None:
        self.df = df

        if post_init:
            self.post_init()

    def post_init(self) -> None:
        """Post-initialization hook for subclasses to override."""
        pass

    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        return len(self.df)

    def __init_subclass__(
        cls,
        *,
        default_invocation: Literal["parquet", "csv", "init", "hf", ""] = "",
        default_split_name: str = "train",
        default_invocation_args: tuple[Any, ...] = (),
        default_invocation_kwargs: dict[str, Any] = {},  # noqa: B006
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        cls.default_invocation = default_invocation
        cls.default_split_name = default_split_name
        cls.default_invocation_args = default_invocation_args
        cls.default_invocation_kwargs = default_invocation_kwargs

    @classmethod
    def create_default(cls) -> Self:
        """Create a default dataset instance as defined by the subclass."""
        match cls.default_invocation:
            case "parquet":
                return cls.from_parquet(*cls.default_invocation_args, **cls.default_invocation_kwargs)
            case "csv":
                return cls.from_csv(*cls.default_invocation_args, **cls.default_invocation_kwargs)
            case "init":
                return cls(*cls.default_invocation_args, **cls.default_invocation_kwargs)
            case "hf":
                return cls.from_hf_dataset(
                    *cls.default_invocation_args,
                    **cls.default_invocation_kwargs,
                )[cls.default_split_name]
            case _:
                raise ValueError("No default set")

    @classmethod
    def create_defaults(cls) -> dict[str, Self]:
        """Create default dataset instances for all splits defined by the subclass."""
        match cls.default_invocation:
            case "parquet" | "csv" | "init":
                return {cls.default_split_name: cls.create_default()}
            case "hf":
                return cls.from_hf_dataset(
                    *cls.default_invocation_args,
                    **cls.default_invocation_kwargs,
                )
            case _:
                raise ValueError("No default set")

    def create_label_map(self) -> dict[str, int]:
        """
        Creates a mapping from category names to unique integer identifiers. Requires the label column to be set.

        Returns:
            A dictionary mapping labels to unique integer IDs.

        See Also:
            - :py:class:`.CategorizerDataset` for an example of a dataset that requires label mapping.
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
            self.__class__(train, post_init=False),
            self.__class__(val, post_init=False),
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

    @classmethod
    def from_parquet(cls, path: str, **kwargs: Any) -> Self:
        """
        Create a PandasDataset instance from a Parquet file.

        Args:
            path: The path to the Parquet file.
            **kwargs: Additional keyword arguments to pass to `pd.read_parquet`.
        """
        return cls(pd.read_parquet(path, **kwargs))

    @classmethod
    def from_hf_dataset(cls, dataset: str, **kwargs: Any) -> dict[str, Self]:
        """
        Create PandasDatasets from a Hugging Face dataset, one for each split.

        Args:
            dataset: The name of the Hugging Face dataset.
            **kwargs: Additional keyword arguments to pass to `datasets.load_dataset`.
        """
        return {k: cls(v.to_pandas()) for k, v in datasets.load_dataset(dataset, **kwargs).items()}
