import torch
import torch.nn as nn
from transformers import Trainer

from ypl.training.data.base import CollateType
from ypl.training.model.routing import RoutingMultilabelClassificationModel


class RoutingMultilabelTrainer(Trainer):  # type: ignore[misc]
    """Transformers trainer for routing multilabel classification models."""

    def compute_loss(
        self, model: RoutingMultilabelClassificationModel, inputs: CollateType, return_outputs: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        logits = model(inputs)
        loss: torch.Tensor = torch.zeros(1, device=self.args.device)
        loss_fn = nn.LogSigmoid()

        for idx, (better_model_id, worse_model_id) in enumerate(
            zip(
                inputs["better_model_labels"],
                inputs["worse_model_labels"],
                strict=False,
            )
        ):
            # Not the most efficient but more readable
            loss += -loss_fn(logits[idx, better_model_id] - logits[idx, worse_model_id])

        loss /= len(inputs["better_model_labels"])

        return (loss, logits) if return_outputs else loss
