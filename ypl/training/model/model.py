import torch
import torch.nn as nn

from ypl.training.torch_utils import DeviceMixin


# For now, assume that all models can fit on a single device.
class YuppModel(DeviceMixin, nn.Module):
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
