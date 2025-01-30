from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ypl.pytorch.data.base import StrTensorDict
from ypl.pytorch.model.base import YuppClassificationModel
from ypl.pytorch.torch_utils import TorchAccelerationMixin
from ypl.utils import dict_extract

OutputType = TypeVar("OutputType")


class CategorizerModel(YuppClassificationModel, Generic[OutputType]):
    """Abstract base class for categorizer models."""

    def categorize(self, prompt: str) -> OutputType:
        """Returns an output label of the prompt."""
        raise NotImplementedError


class HFCategorizerModel(TorchAccelerationMixin, CategorizerModel[OutputType]):
    """Classification model for categorizer."""

    def __init__(
        self,
        model_name: str,
        label_map: dict[str, int],
        multilabel: bool = False,
        multilabel_threshold: float = 0.55,  # seems to work well empirically
    ):
        """
        Initialize the classification model.

        Args:
            model_name: The name of the pretrained model.
            label_map: Mapping from category names to unique integer IDs.
        """
        super().__init__(model_name=model_name, label_map=label_map, multilabel=multilabel)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.category2id = label_map
        self.id2category = {v: k for k, v in label_map.items()}
        self.multilabel_threshold = multilabel_threshold

    def postprocess_single_label_output(self, outputs: torch.Tensor) -> OutputType:
        raise NotImplementedError

    def postprocess_multilabel_output(self, outputs: torch.Tensor) -> OutputType:
        raise NotImplementedError

    @torch.no_grad()
    async def acategorize(self, prompt: str) -> OutputType:
        self.eval()
        input_ids = self.to_alike(
            self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            )
        )

        attention_mask = torch.ones_like(input_ids)

        if self.is_dynamo_compiled or self.is_cuda_graph_compiled:
            # Pad the input to the nearest multiple of 64
            padding = (64 - input_ids.shape[-1] % 64) % 64
            input_ids = torch.nn.functional.pad(input_ids, (0, padding))
            attention_mask = torch.nn.functional.pad(attention_mask, (0, padding))

            # Use self() if compiled:
            if self.is_dynamo_compiled:
                outputs = self(dict(input_ids=input_ids, attention_mask=attention_mask))["logits"]
            else:
                outputs = (await self.acuda_graph_forward(dict(input_ids=input_ids, attention_mask=attention_mask)))[
                    "logits"
                ]
        else:
            outputs = self(dict(input_ids=input_ids, attention_mask=attention_mask))["logits"]

        return (
            self.postprocess_multilabel_output(outputs)
            if self.multilabel
            else self.postprocess_single_label_output(outputs)
        )

    @torch.no_grad()
    def categorize(self, prompt: str) -> OutputType:
        self.eval()
        input_ids = self.to_alike(
            self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            )
        )

        attention_mask = torch.ones_like(input_ids)

        if self.is_dynamo_compiled or self.is_cuda_graph_compiled:
            # Pad the input to the nearest multiple of 64
            padding = (64 - input_ids.shape[-1] % 64) % 64
            input_ids = torch.nn.functional.pad(input_ids, (0, padding))
            attention_mask = torch.nn.functional.pad(attention_mask, (0, padding))

            # Use self() if compiled:
            if self.is_dynamo_compiled:
                outputs = self(dict(input_ids=input_ids, attention_mask=attention_mask))["logits"]
            else:
                outputs = self.cuda_graph_forward(dict(input_ids=input_ids, attention_mask=attention_mask))["logits"]

        else:
            outputs = self(dict(input_ids=input_ids, attention_mask=attention_mask))["logits"]

        return (
            self.postprocess_multilabel_output(outputs)
            if self.multilabel
            else self.postprocess_single_label_output(outputs)
        )

    @property
    def _warmup_inputs(self) -> list[StrTensorDict]:
        inputs: list[StrTensorDict] = []
        chunk_size_start = 64
        chunk_size_step = chunk_size_start

        for chunk_size in range(chunk_size_start, 512 + 1, chunk_size_step):
            inputs.append(
                dict(
                    input_ids=self.to_alike(torch.randint(0, 100, (1, chunk_size))),
                    attention_mask=self.to_alike(torch.ones((1, chunk_size), dtype=torch.long)),
                )
            )

        return inputs

    @property
    def _dynamo_options(self) -> dict[str, Any]:
        return {"mode": "reduce-overhead", "dynamic": False, "fullgraph": True}

    def forward(self, batch: StrTensorDict) -> StrTensorDict:
        """Perform a forward pass through the model to obtain logits."""
        raise NotImplementedError

    def _save_pretrained(self, save_directory: str) -> None:  # type: ignore
        raise NotImplementedError

    @classmethod
    def _from_pretrained(cls, *, model_id: str, **kwargs: Any) -> "HFCategorizerModel":
        raise NotImplementedError


class PromptTopicDifficultyModel(HFCategorizerModel[tuple[str, int] | tuple[list[str], int]]):
    """Classification model for categorizer."""

    def __init__(self, model_name: str, label_map: dict[str, int], **kwargs: Any) -> None:
        super().__init__(model_name, label_map, **kwargs)
        self.category_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_map))
        self.difficulty_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=20)

    def postprocess_multilabel_output(self, outputs: torch.Tensor) -> tuple[list[str], int]:
        category_outputs = outputs[:, : len(self.category2id)]
        difficulty_outputs = outputs[:, len(self.category2id) :]

        category_ids = category_outputs.sigmoid().squeeze().tolist()
        category_ids = [i for i, v in enumerate(category_ids) if v > self.multilabel_threshold]

        return [self.id2category[i] for i in category_ids], int(difficulty_outputs.squeeze().argmax().item() + 1)

    def postprocess_single_label_output(self, outputs: torch.Tensor) -> tuple[str, int]:
        category_outputs = outputs[:, : len(self.category2id)]
        difficulty_outputs = outputs[:, len(self.category2id) :]

        category_id = int(category_outputs.argmax().item())
        difficulty_id = int(difficulty_outputs.argmax().item())

        return self.id2category[category_id], difficulty_id + 1

    def _save_pretrained(self, save_directory: str) -> None:  # type: ignore
        Path(save_directory, "base_model").write_text(self.category_model.config._name_or_path)
        torch.save(self.multilabel, Path(save_directory) / "is_multilabel.pt")
        torch.save(self.category_model.state_dict(), Path(save_directory) / "category_model.bin")
        torch.save(self.difficulty_model.state_dict(), Path(save_directory) / "difficulty_model.bin")
        torch.save(self.category2id, Path(save_directory) / "category_map.pt")

    @classmethod
    def _from_pretrained(cls, *, model_id: str, **kwargs: Any) -> "PromptTopicDifficultyModel":
        base_model_id = Path(model_id, "base_model").read_text()
        category_map = torch.load(Path(model_id) / "category_map.pt")

        try:
            is_multilabel = torch.load(Path(model_id) / "is_multilabel.pt")
        except FileNotFoundError:
            is_multilabel = False

        model = cls(base_model_id, category_map, multilabel=is_multilabel)
        model.category_model.load_state_dict(torch.load(Path(model_id) / "category_model.bin"))
        model.difficulty_model.load_state_dict(torch.load(Path(model_id) / "difficulty_model.bin"))

        return model

    def forward(self, batch: StrTensorDict) -> StrTensorDict:
        clogs = self.category_model(**dict_extract(batch, {"input_ids", "attention_mask"})).logits
        dlogs = self.difficulty_model(**dict_extract(batch, {"input_ids", "attention_mask"})).logits

        return dict(logits=torch.cat([clogs, dlogs], dim=-1))


class OnlinePromptClassifierModel(HFCategorizerModel[str]):
    """Classification model for categorizer."""

    def __init__(self, model_name: str, label_map: dict[str, int], **kwargs: Any) -> None:
        super().__init__(model_name, label_map, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_map))

    def postprocess_single_label_output(self, outputs: torch.Tensor) -> str:
        category_id = int(outputs.argmax().item())
        return self.id2category[category_id]

    def _save_pretrained(self, save_directory: str) -> None:  # type: ignore
        Path(save_directory, "base_model").write_text(self.model.config._name_or_path)
        torch.save(self.model.state_dict(), Path(save_directory) / "model.bin")
        torch.save(self.category2id, Path(save_directory) / "category_map.pt")

    @classmethod
    def _from_pretrained(cls, *, model_id: str, **kwargs: Any) -> "OnlinePromptClassifierModel":
        base_model_id = Path(model_id, "base_model").read_text()
        category_map = torch.load(Path(model_id) / "category_map.pt")

        model = cls(base_model_id, category_map)
        model.model.load_state_dict(torch.load(Path(model_id) / "model.bin"))

        return model

    def forward(self, batch: StrTensorDict) -> StrTensorDict:
        return dict(logits=self.model(**dict_extract(batch, {"input_ids", "attention_mask"})).logits)
