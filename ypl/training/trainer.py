import torch
import torch.nn as nn
import torch.utils.data as tud
from transformers import Trainer

from ypl.training.data.base import CollateType
from ypl.training.data.routing import RoutingDataset
from ypl.training.model.routing import RoutingModel, RoutingMultilabelClassificationModel


class RoutingMultilabelTrainer(Trainer):  # type: ignore[misc]
    """Transformers trainer for routing multilabel classification models."""

    def compute_loss(
        self, model: RoutingMultilabelClassificationModel, inputs: CollateType, return_outputs: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        logits = model(inputs)
        normalizer = nn.LogSoftmax(dim=-1)
        loss: torch.Tensor = torch.zeros(1, device=self.args.device)
        logprobs = normalizer(logits)

        for idx, model_ids in enumerate(inputs["model_labels"]):
            loss += -logprobs[idx, model_ids[0]]

        loss /= len(inputs["model_labels"])
        loss = loss.squeeze()

        return (loss, logprobs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: tud.Dataset | dict[str, tud.Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        self.model.eval()

        assert isinstance(eval_dataset, RoutingDataset), "eval_dataset must be an instance of RoutingDataset"
        assert isinstance(self.model, RoutingModel), "model must be an instance of RoutingModel"

        accuracy = []

        for example in eval_dataset:  # type: ignore[attr-defined]
            scores = self.model.route_to_models(example.prompt)
            accuracy.append(int(list(scores).index(example.models[0]) == 0))

        return dict(accuracy=sum(accuracy) / len(accuracy))
