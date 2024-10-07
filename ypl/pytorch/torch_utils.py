from typing import Self, TypeVar

import numpy as np
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


class PyTorchRNGMixin:
    """
    Mixin class to add random number generators to a class.
    """

    _np_rng: np.random.RandomState | None = None
    _torch_rng: torch.Generator | None = None
    _seed: int | None = None

    def set_seed(self, seed: int) -> None:
        if self._seed is not None:
            raise ValueError("Seed already set")

        self._seed = seed

    def set_deterministic(self, deterministic: bool) -> None:
        # This is a global setting...
        torch.use_deterministic_algorithms(deterministic)

    def with_seed(self, seed: int) -> Self:
        self.set_seed(seed)
        return self

    def get_numpy_rng(self) -> np.random.RandomState:
        if self._np_rng is None:
            self._np_rng = np.random.RandomState(self._seed)

        return self._np_rng

    def get_torch_rng(self) -> torch.Generator:
        if self._torch_rng is None:
            self._torch_rng = torch.Generator()

            if self._seed is not None:
                self._torch_rng.manual_seed(self._seed)

        return self._torch_rng
