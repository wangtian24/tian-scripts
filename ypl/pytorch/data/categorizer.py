import ast
from collections import Counter

import numpy as np
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer

from ypl.pytorch.data.base import PandasDataset, StrTensorDict, TokenizerCollator


class CategorizerTrainingExample(BaseModel):
    """
    Represents a single categorizer example containing a prompt and the corresponding category and difficulty.
    """

    prompt: str
    category: list[str] | str
    difficulty: int | None = None  # an integer between 1 and 10


class CategorizerDataset(PandasDataset[CategorizerTrainingExample]):
    """
    Dataset class for categorizer examples. This class handles the creation of category and difficulty mappings
    and categorizer examples.
    """

    label_column: str = "category"

    def __getitem__(self, index: int) -> CategorizerTrainingExample:
        """
        Retrieves the categorizer example at the specified index.

        Args:
            index: The index of the categorizer example to retrieve.

        Returns:
            The categorizer example at the given index.
        """

        def remap(x: int) -> int:
            if np.isnan(x):
                return 0

            return x

        row = self.df.iloc[index]

        try:
            category = ast.literal_eval(row["category"])
        except:  # noqa: E722
            category = row["category"]

        category = category if isinstance(category, str | list) else ""
        prompt = row["prompt"] if isinstance(row["prompt"], str) else ""

        return CategorizerTrainingExample(
            prompt=prompt,
            category=category,
            difficulty=remap(row["difficulty"]) if "difficulty" in row else None,
        )

    def compute_label_pos_weights(self, label_map: dict[str, int], p: float = 0.4) -> torch.Tensor:
        """
        Computes the positive label weights for the categorizer model on the number of examples per category.
        Takes an inverse power of the proportion of examples per category to downweight the majority classes.
        Empirically, using a power of 0.4 works well.

        Args:
            label_map: Mapping from category names to unique integer IDs.
            p: The power of the inverse to use.

        Returns:
            A tensor of positive label weights for use with `BCEWithLogitsLoss`.
        """
        label_counts: dict[int, int] = Counter()

        for _, row in self.df.iterrows():
            try:
                categories = ast.literal_eval(row["category"])
            except:  # noqa: E722
                continue

            if not isinstance(categories, list):  # only for multilabel datasets
                continue

            for category in categories:
                label_counts[label_map[category]] += 1

        counts = torch.zeros(len(label_map), dtype=torch.float)

        for label, count in label_counts.items():
            counts[label] = count

        counts /= counts.sum()  # normalize to sum to 1

        return (1 / counts) ** p  # type: ignore[no-any-return]


class CategorizerCollator(TokenizerCollator[CategorizerTrainingExample]):
    """
    Collator for categorizer examples that prepares batches for model input. This collator encodes prompts and
    maps categories to their corresponding indices.
    """

    def __init__(self, tokenizer: AutoTokenizer, label_map: dict[str, int], multilabel: bool = False) -> None:
        super().__init__(tokenizer)
        self.label_map = label_map
        self.multilabel = multilabel

    def collate(self, batch: list[CategorizerTrainingExample]) -> StrTensorDict:
        """
        Collates a batch of categorizer examples into tensors suitable for model input. This method encodes
        the prompts and converts categories into tensor formats.

        Args:
            batch: A list of categorizer examples to collate.

        Returns:
            A dictionary containing tokenized inputs and model label tensors.
        """
        tokenizer_output = self.tokenizer.batch_encode_plus(
            [example.prompt for example in batch],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_attention_mask=True,
        ).data

        if self.multilabel:
            category_labels = torch.zeros(len(batch), len(self.label_map))

            for i, example in enumerate(batch):
                for category in example.category:
                    category_labels[i, self.label_map[category]] = 1

            tokenizer_output["category_labels"] = category_labels
        else:
            tokenizer_output["category_labels"] = torch.tensor(
                [self.label_map[example.category] for example in batch]  # type: ignore[index]
            )

        if all(example.difficulty is not None for example in batch):
            tokenizer_output["difficulty_labels"] = torch.tensor(
                [
                    example.difficulty - 1  # type: ignore[operator]
                    for example in batch
                ]
            )

        return tokenizer_output  # type: ignore
