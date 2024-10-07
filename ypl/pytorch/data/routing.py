import pandas as pd
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer

from ypl.pytorch.data.base import CollateType, PandasDataset, TokenizerCollator


class RoutingTrainingExample(BaseModel):
    """
    Represents a single routing example containing a prompt and the corresponding ranked models,
    in order from best to worst.
    """

    prompt: str
    models: list[str]


class RoutingDataset(PandasDataset[RoutingTrainingExample]):
    """
    Dataset class for routing examples, extending PandasDataset. This class handles the creation of model mappings and
    retrieval of routing examples.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)

    def create_label_map(self) -> dict[str, int]:
        """
        Creates a mapping from model names to unique integer identifiers.

        Returns:
            A dictionary mapping model names to unique integer IDs.
        """
        all_models = list(dict.fromkeys([x for y in self.df["models"].apply(eval) for x in y]))
        model_map = {model: i for i, model in enumerate(all_models)}

        return model_map

    def __getitem__(self, index: int) -> RoutingTrainingExample:
        """
        Retrieves the routing example at the specified index.

        Args:
            index: The index of the routing example to retrieve.

        Returns:
            The routing example at the given index.
        """
        row = self.df.iloc[index]
        models = eval(row["models"])

        if "gpt-4o" == models[0]:
            # swap gpt-4o with second element
            models[0], models[1] = models[1], models[0]

        return RoutingTrainingExample(prompt=row["prompt"], models=models)


class RoutingCollator(TokenizerCollator[RoutingTrainingExample]):
    """
    Collator for routing examples that prepares batches for model input. This collator encodes prompts and
    maps model labels to their corresponding indices.
    """

    def __init__(self, tokenizer: AutoTokenizer, label_map: dict[str, int]) -> None:
        super().__init__(tokenizer)
        self.label_map = label_map

    def collate(self, batch: list[RoutingTrainingExample]) -> CollateType:
        """
        Collates a batch of routing examples into tensors suitable for model input. This method encodes
        the prompts and converts model labels into tensor formats.

        Args:
            batch: A list of routing examples to collate.

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

        tokenizer_output["model_labels"] = torch.tensor(
            [[self.label_map[model] for model in example.models] for example in batch]
        )

        return tokenizer_output  # type: ignore
