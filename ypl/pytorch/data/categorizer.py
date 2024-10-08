import pandas as pd
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer

from ypl.pytorch.data.base import PandasDataset, StrTensorDict, TokenizerCollator


class CategorizerTrainingExample(BaseModel):
    """
    Represents a single categorizer example containing a prompt and the corresponding category and difficulty.
    """

    prompt: str
    category: str
    difficulty: int  # an integer between 1 and 10


class CategorizerDataset(PandasDataset[CategorizerTrainingExample]):
    """
    Dataset class for categorizer examples. This class handles the creation of category and difficulty mappings
    and categorizer examples.
    """

    label_column: str = "category"

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)

    def __getitem__(self, index: int) -> CategorizerTrainingExample:
        """
        Retrieves the categorizer example at the specified index.

        Args:
            index: The index of the categorizer example to retrieve.

        Returns:
            The categorizer example at the given index.
        """
        row = self.df.iloc[index]
        return CategorizerTrainingExample(prompt=row["prompt"], category=row["category"], difficulty=row["difficulty"])


class CategorizerCollator(TokenizerCollator[CategorizerTrainingExample]):
    """
    Collator for categorizer examples that prepares batches for model input. This collator encodes prompts and
    maps categories to their corresponding indices.
    """

    def __init__(self, tokenizer: AutoTokenizer, label_map: dict[str, int]) -> None:
        super().__init__(tokenizer)
        self.label_map = label_map

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

        tokenizer_output["category_labels"] = torch.tensor([self.label_map[example.category] for example in batch])
        tokenizer_output["difficulty_labels"] = torch.tensor([example.difficulty - 1 for example in batch])

        return tokenizer_output  # type: ignore
