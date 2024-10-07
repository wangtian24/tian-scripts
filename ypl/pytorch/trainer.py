import torch
import torch.nn as nn
import torch.utils.data as tud
from transformers import Trainer

from ypl.pytorch.data.base import StrAnyDict
from ypl.pytorch.data.categorizer import CategorizerDataset
from ypl.pytorch.data.routing import RoutingDataset
from ypl.pytorch.model.categorizer import CategorizerClassificationModel
from ypl.pytorch.model.routing import RoutingModel, RoutingMultilabelClassificationModel


class RoutingMultilabelTrainer(Trainer):  # type: ignore[misc]
    """Transformers trainer for routing multilabel classification models."""

    def compute_loss(
        self, model: RoutingMultilabelClassificationModel, inputs: StrAnyDict, return_outputs: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        logits = model(inputs)["logits"]
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
            print(accuracy[-1], scores)

        return dict(accuracy=sum(accuracy) / len(accuracy))


class CategorizerTrainer(Trainer):  # type: ignore[misc]
    """Transformers trainer for categorizer models."""

    def compute_loss(
        self, model: CategorizerClassificationModel, inputs: StrAnyDict, return_outputs: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        logits = model(inputs)["logits"]
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits[:, :-10], inputs["category_labels"]) + loss_fn(
            logits[:, -10:], inputs["difficulty_labels"]
        )

        return (loss, logits) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: tud.Dataset | dict[str, tud.Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        self.model.eval()

        assert isinstance(eval_dataset, CategorizerDataset), "eval_dataset must be an instance of CategorizerDataset"
        assert isinstance(
            self.model, CategorizerClassificationModel
        ), "model must be an instance of CategorizerClassificationModel"

        cat_accuracy = []
        diff_accuracy = []

        for example in eval_dataset:  # type: ignore[attr-defined]
            category, difficulty = self.model.categorize(example.prompt)
            cat_accuracy.append(int(category == example.category))
            diff_accuracy.append(int(difficulty == example.difficulty))
            print(cat_accuracy[-1], diff_accuracy[-1], category, difficulty, example.prompt[:100])

        return dict(
            cat_accuracy=sum(cat_accuracy) / len(cat_accuracy),
            diff_accuracy=sum(diff_accuracy) / len(diff_accuracy),
        )
