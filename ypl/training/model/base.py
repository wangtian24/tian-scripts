import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from ypl.training.torch_utils import DeviceMixin


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
