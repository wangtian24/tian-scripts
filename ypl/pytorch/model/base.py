import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from ypl.pytorch.data.base import StrAnyDict
from ypl.pytorch.torch_utils import DeviceMixin


# For now, assume that all models can fit on a single device.
class YuppModel(PyTorchModelHubMixin, DeviceMixin, nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()

    @property
    def device(self) -> torch.device:
        self._device = next(self.parameters()).device
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self.to(device)
        self._device = device

    def forward(self, batch: StrAnyDict) -> StrAnyDict:
        raise NotImplementedError


class YuppClassificationModel(YuppModel):
    def __init__(self, model_name: str, label_map: dict[str, int]) -> None:
        super().__init__()
        self.model_name = model_name
        self.label_map = label_map
