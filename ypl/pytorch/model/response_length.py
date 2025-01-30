import asyncio
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ypl.pytorch.data.base import StrTensorDict
from ypl.pytorch.model.base import YuppBucketingModel
from ypl.pytorch.torch_utils import TorchAccelerationMixin
from ypl.utils import dict_extract


class ResponseLengthModel(YuppBucketingModel):
    """Abstract base class for response length models."""

    def predict_length(self, prompt: str) -> int:
        """Returns the bucket of the anticipated response length of the prompt."""
        raise NotImplementedError


class TransformerResponseLengthModel(TorchAccelerationMixin, ResponseLengthModel):
    """Transformer model for response length prediction."""

    def __init__(self, model_name: str, buckets: list[int]):
        """
        Initialize the response length model.

        Args:
            model_name: The name of the pretrained model.
            buckets: Mapping from category names to unique integer IDs.
        """
        super().__init__(model_name=model_name, buckets=buckets)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(buckets))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @torch.no_grad()
    async def apredict_length(self, prompt: str) -> int:
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
            outputs = self.model(input_ids, attention_mask=attention_mask).logits.squeeze()

        bucket_id = outputs.argmax().item()

        return bucket_id  # type: ignore[no-any-return]

    @torch.no_grad()
    def predict_length(self, prompt: str) -> int:
        return asyncio.run(self.apredict_length(prompt))

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
        logs = self.model(**dict_extract(batch, {"input_ids", "attention_mask"})).logits

        return dict(logits=logs)

    def _save_pretrained(self, save_directory: str) -> None:  # type: ignore
        Path(save_directory, "base_model").write_text(self.model.config._name_or_path)
        torch.save(self.model.state_dict(), Path(save_directory) / "model.bin")
        torch.save(self.buckets, Path(save_directory) / "buckets.pt")

    @classmethod
    def _from_pretrained(cls, *, model_id: str, **kwargs: Any) -> "TransformerResponseLengthModel":
        base_model_id = Path(model_id, "base_model").read_text()
        buckets = torch.load(Path(model_id) / "buckets.pt")
        model = cls(base_model_id, buckets)
        model.model.load_state_dict(torch.load(Path(model_id) / "model.bin"))

        return model
