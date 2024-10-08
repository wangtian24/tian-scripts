from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ypl.pytorch.data.base import StrTensorDict
from ypl.pytorch.model.base import YuppClassificationModel
from ypl.utils import dict_extract


class RoutingModel(YuppClassificationModel):
    """Abstract base class for routing models."""

    def route_to_models(self, prompt: str) -> dict[str, float]:
        """Returns a dictionary ordered by the scores of the models for answering the prompt."""
        raise NotImplementedError


class RoutingMultilabelClassificationModel(RoutingModel):
    """Multilabel classification model for routing."""

    def __init__(self, model_name: str, label_map: dict[str, int]):
        """
        Initialize the multilabel classification model.

        Args:
            model_name: The name of the pretrained model.
            label_map: Mapping from model names to unique integer IDs.
        """
        super().__init__(model_name=model_name, label_map=label_map)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_map))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model2id = label_map
        self.id2model = {v: k for k, v in label_map.items()}

    @torch.no_grad()
    def route_to_models(self, prompt: str) -> dict[str, float]:
        input_ids = self.to_alike(
            self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            )
        )
        outputs = self.model(input_ids).logits
        output_logits = outputs.squeeze().softmax(dim=0).cpu().numpy()

        return {
            k: v
            for k, v in sorted(
                [(self.id2model[i], score) for i, score in enumerate(output_logits)],
                key=lambda x: x[1],
                reverse=True,
            )
        }

    def forward(self, batch: StrTensorDict) -> StrTensorDict:
        """
        Perform a forward pass through the model to obtain logits. The logits are treated as multilabel classification
        scores for each model, i.e., to get the scores, an elementwise sigmoid should be computed.

        Args:
            batch: A batch of input data containing 'input_ids' and 'attention_mask'.

        Returns:
            The logits output by the model.
        """
        return dict(logits=self.model(**dict_extract(batch, {"input_ids", "attention_mask"})).logits)

    def _save_pretrained(self, save_directory: str) -> None:
        Path(save_directory, "base_model").write_text(self.model.config._name_or_path)
        torch.save(self.model.state_dict(), Path(save_directory) / "pytorch_model.bin")
        torch.save(self.model2id, Path(save_directory) / "model_map.pt")

    @classmethod
    def _from_pretrained(cls, *, model_id: str, **kwargs: Any) -> "RoutingMultilabelClassificationModel":
        base_model_id = Path(model_id, "base_model").read_text()
        model2id = torch.load(Path(model_id) / "model_map.pt")
        model = cls(base_model_id, model2id)
        model.model.load_state_dict(torch.load(Path(model_id) / "pytorch_model.bin"))

        return model
