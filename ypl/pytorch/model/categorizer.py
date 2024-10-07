from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ypl.pytorch.data.base import CollateType
from ypl.pytorch.model.base import YuppClassificationModel
from ypl.utils import dict_extract


class CategorizerModel(YuppClassificationModel):
    """Abstract base class for categorizer models."""

    def categorize(self, prompt: str) -> tuple[str, int]:
        """Returns a tuple of the category and difficulty of the prompt."""
        raise NotImplementedError


class CategorizerClassificationModel(CategorizerModel):
    """Classification model for categorizer."""

    def __init__(self, model_name: str, label_map: dict[str, int]):
        """
        Initialize the classification model.

        Args:
            model_name: The name of the pretrained model.
            label_map: Mapping from category names to unique integer IDs.
        """
        super().__init__(model_name=model_name, label_map=label_map)
        self.category_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_map))
        self.difficulty_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=10)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.category2id = label_map
        self.id2category = {v: k for k, v in label_map.items()}

    @torch.no_grad()
    def categorize(self, prompt: str) -> tuple[str, int]:
        input_ids = self.to_alike(
            self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            )
        )
        category_outputs = self.category_model(input_ids).logits.squeeze()
        difficulty_outputs = self.difficulty_model(input_ids).logits.squeeze()

        category_id = category_outputs.argmax().item()
        difficulty_id = difficulty_outputs.argmax().item()

        return self.id2category[category_id], difficulty_id + 1

    def forward(self, batch: CollateType) -> torch.Tensor:
        """
        Perform a forward pass through the model to obtain logits. The logits are treated as multilabel classification
        scores for each model, i.e., to get the scores, an elementwise sigmoid should be computed.

        Args:
            batch: A batch of input data containing 'input_ids' and 'attention_mask'.

        Returns:
            The logits output by the model.
        """
        clogs = self.category_model(**dict_extract(batch, {"input_ids", "attention_mask"})).logits
        dlogs = self.difficulty_model(**dict_extract(batch, {"input_ids", "attention_mask"})).logits

        return torch.cat([clogs, dlogs], dim=-1)

    def _save_pretrained(self, save_directory: str) -> None:
        Path(save_directory, "base_model").write_text(self.category_model.config._name_or_path)
        torch.save(self.category_model.state_dict(), Path(save_directory) / "category_model.bin")
        torch.save(self.difficulty_model.state_dict(), Path(save_directory) / "difficulty_model.bin")
        torch.save(self.category2id, Path(save_directory) / "category_map.pt")

    @classmethod
    def _from_pretrained(cls, *, model_id: str, **kwargs: Any) -> "CategorizerClassificationModel":
        base_model_id = Path(model_id, "base_model").read_text()
        category_map = torch.load(Path(model_id) / "category_map.pt")
        model = cls(base_model_id, category_map)
        model.category_model.load_state_dict(torch.load(Path(model_id) / "category_model.bin"))
        model.difficulty_model.load_state_dict(torch.load(Path(model_id) / "difficulty_model.bin"))

        return model
