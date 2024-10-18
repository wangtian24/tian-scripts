from collections import defaultdict
from typing import no_type_check

import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.utils.data as tud
from scipy.stats import spearmanr
from transformers import Trainer

from ypl.pytorch.data.base import StrTensorDict
from ypl.pytorch.data.categorizer import CategorizerDataset
from ypl.pytorch.data.prompting import ResponseLengthDataset
from ypl.pytorch.data.routing import RoutingDataset
from ypl.pytorch.model.categorizer import CategorizerClassificationModel
from ypl.pytorch.model.response_length import ResponseLengthModel
from ypl.pytorch.model.routing import RoutingModel, RoutingMultilabelClassificationModel


class ResponseLengthTrainer(Trainer):  # type: ignore[misc]
    """Transformers trainer for response length models."""

    def compute_loss(
        self, model: ResponseLengthModel, inputs: StrTensorDict, return_outputs: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        logits = model(inputs)["logits"]
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, inputs["response_lengths"])

        return (loss, logits) if return_outputs else loss

    @no_type_check
    def evaluate(
        self,
        eval_dataset: tud.Dataset | dict[str, tud.Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        self.model.eval()

        assert isinstance(
            eval_dataset, ResponseLengthDataset
        ), "eval_dataset must be an instance of ResponseLengthDataset"
        assert isinstance(self.model, ResponseLengthModel), "model must be an instance of ResponseLengthModel"

        accuracy = []
        preds = []
        truths = []

        for example in eval_dataset:
            truth = self.model.bucket_index(example.response_length)
            pred = self.model.predict_length(example.prompt)
            accuracy.append(int(pred == truth))
            preds.append(pred)
            truths.append(truth)

        return dict(accuracy=sum(accuracy) / len(accuracy), rho=spearmanr(truths, preds).correlation)


class RoutingMultilabelTrainer(Trainer):  # type: ignore[misc]
    """Transformers trainer for routing multilabel classification models."""

    def compute_loss(
        self, model: RoutingMultilabelClassificationModel, inputs: StrTensorDict, return_outputs: bool = False
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
            print(sum(accuracy) / len(accuracy))

        return dict(accuracy=sum(accuracy) / len(accuracy))


class CategorizerTrainer(Trainer):  # type: ignore[misc]
    """Transformers trainer for categorizer models."""

    def compute_loss(
        self, model: CategorizerClassificationModel, inputs: StrTensorDict, return_outputs: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        logits = model(inputs)["logits"]

        if model.multilabel:
            loss_fn_multilabel = nn.BCEWithLogitsLoss()
            loss_fn_difficulty = nn.CrossEntropyLoss()
            loss = loss_fn_multilabel(logits[:, :-10], inputs["category_labels"]) + loss_fn_difficulty(
                logits[:, -10:], inputs["difficulty_labels"]
            )
        else:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fn(logits[:, :-10], inputs["category_labels"]) + loss_fn(
                logits[:, -10:], inputs["difficulty_labels"]
            )

        return (loss, logits) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: tud.Dataset | dict[str, tud.Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
        multilabel: bool = False,
    ) -> dict[str, float]:
        self.model.eval()

        assert isinstance(eval_dataset, CategorizerDataset), "eval_dataset must be an instance of CategorizerDataset"
        assert isinstance(
            self.model, CategorizerClassificationModel
        ), "model must be an instance of CategorizerClassificationModel"

        cat_metric = defaultdict(list)
        diff_accuracy = []

        for example in eval_dataset:  # type: ignore[attr-defined]
            category, difficulty = self.model.categorize(example.prompt)

            if multilabel:
                assert isinstance(example.category, list) and isinstance(category, list)
                ohv_enc = pd.get_dummies(list(set(example.category + category)))  # type: ignore[arg-type]
                cat_metric["f1"].append(
                    sklearn.metrics.f1_score(
                        ohv_enc[example.category].values.astype(int).sum(-1),
                        ohv_enc[category].values.astype(int).sum(-1),
                        average="micro",
                    )
                )
                cat_metric["exact_match"].append(int(set(category) == set(example.category)))
            else:
                cat_metric["accuracy"].append(int(category == example.category))

            if example.difficulty != 0:
                diff_accuracy.append(int(difficulty == example.difficulty))

        return dict(
            diff_accuracy=sum(diff_accuracy) / len(diff_accuracy),
            **{f"cat_{k}": sum(v) / len(v) for k, v in cat_metric.items()},
        )
