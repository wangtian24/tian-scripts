from typing import TypeVar

import torch

T = TypeVar("T")


class DeviceMixin:
    """Mixin for associating a single device with a class."""

    _device: torch.device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device

    def to_alike(self, input_: T) -> T:
        """
        Recursively move the input to the device of the current class.

        Args:
            input_: The input to move to the device of the current class. Can be a tensor, list, tuple, or
                dict. Instances other than these result in a no-op.

        Returns:
            The input moved to the device of the current class.
        """
        match input_:
            case torch.Tensor():
                return input_.to(self.device)  # type: ignore
            case list():
                return [self.to_alike(item) for item in input_]  # type: ignore
            case tuple():
                return tuple(self.to_alike(item) for item in input_)  # type: ignore
            case dict():
                return {k: self.to_alike(v) for k, v in input_.items()}  # type: ignore
            case DeviceMixin():
                return input_.to(self.device)  # type: ignore
            case _:
                return input_
