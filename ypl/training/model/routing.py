import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ypl.training.data.base import CollateType
from ypl.training.model.model import YuppModel
from ypl.utils import dict_extract


class RoutingModel(YuppModel):
    """Abstract base class for routing models."""

    def route_to_models(self, prompt: str) -> list[tuple[str, float]]:
        """Returns a list of model-score tuples sorted by the score of the model being the better model."""
        raise NotImplementedError


class RoutingMultilabelClassificationModel(RoutingModel):
    """Multilabel classification model for routing."""

    def __init__(self, model_name: str, model_map: dict[str, int]):
        """
        Initialize the multilabel classification model.

        Args:
            model_name: The name of the pretrained model.
            model_map: Mapping from model names to unique integer IDs.
        """
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(model_map),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model2id = model_map
        self.id2model = {v: k for k, v in model_map.items()}

    @torch.no_grad()
    def route_to_models(self, prompt: str) -> list[tuple[str, float]]:
        """
        Route the given prompt to models based on their computed scores.

        Args:
            prompt: The input text prompt.

        Returns:
            list[tuple[str, float]]: A sorted list of (model name, score) tuples in descending order of scores.
        """
        input_ids = self.to_alike(self.tokenizer.encode(prompt, return_tensors="pt"))
        outputs = self.model(input_ids).logits
        output_logits = outputs.squeeze().softmax(dim=0).cpu().numpy()

        return sorted(
            [(self.id2model[i], score) for i, score in enumerate(output_logits)],
            key=lambda x: x[1],
            reverse=True,
        )

    def forward(self, batch: CollateType) -> torch.Tensor:
        """
        Perform a forward pass through the model to obtain logits. The logits are treated as multilabel classification
        scores for each model, i.e., to get the scores, an elementwise sigmoid should be computed.

        Args:
            batch: A batch of input data containing 'input_ids' and 'attention_mask'.

        Returns:
            The logits output by the model.
        """
        return self.model(**dict_extract(batch, {"input_ids", "attention_mask"})).logits  # type: ignore
