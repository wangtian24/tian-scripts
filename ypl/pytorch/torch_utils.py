import logging
from typing import Any, Self, TypeVar

import numpy as np
import torch

from ypl.pytorch.serve.cuda_graph import CudaGraphFunctor, CudaGraphPoolExecutor

logging.basicConfig(level=logging.INFO)
T = TypeVar("T")
StrTensorDict = dict[str, torch.Tensor]


class DeviceMixin:
    """Mixin for associating a single device with a class."""

    _device: torch.device = torch.device("cpu")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        return super().__init_subclass__(**kwargs)

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

    def __init_subclass__(cls, **kwargs: Any) -> None:
        return super().__init_subclass__(**kwargs)

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


class TorchAccelerationMixin:
    """Mixin for adding inference acceleration routines to a model, such as TorchDynamo and CUDA graphs."""

    _is_dynamo_compiled: bool = False
    _is_cuda_graph_compiled: bool = False
    _dynamo_cache_size_limit: int = 128
    _cuda_graph_executor: CudaGraphPoolExecutor | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        from ypl.pytorch.model.base import YuppModel

        if not issubclass(cls, YuppModel):
            raise TypeError("TorchDynamoMixin must be used together with YuppModel")

        return super().__init_subclass__(**kwargs)

    @torch.no_grad()
    def compile_cuda_graphs(self, num_graphs_per_input: int = 4) -> None:
        from ypl.pytorch.model.base import YuppModel

        assert isinstance(self, YuppModel)

        self.eval()
        graph_functors = []

        for input in self._warmup_inputs:
            for _ in range(num_graphs_per_input):
                logging.info(f"Compiling CUDA graph {[v.size() for v in input.values()]}")
                g = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                self(input)  # warmup
                torch.cuda.synchronize()

                static_input = {k: v.clone().detach() for k, v in input.items()}
                graph_functor = CudaGraphFunctor(g, static_input, s)
                torch.cuda.synchronize()

                with graph_functor.capture():
                    graph_functor.set_output(self(static_input))

                torch.cuda.synchronize()
                graph_functors.append(graph_functor)

        self._cuda_graph_executor = CudaGraphPoolExecutor(graph_functors)  # type: ignore[assignment]
        self._is_cuda_graph_compiled = True  # type: ignore[assignment]

    def cuda_graph_forward(self, input: StrTensorDict) -> StrTensorDict:
        return self.cuda_graph_executor.submit(input)

    async def acuda_graph_forward(self, input: StrTensorDict) -> StrTensorDict:
        return await self.cuda_graph_executor.asubmit(input)

    @property
    def is_dynamo_compiled(self) -> bool:
        return self._is_dynamo_compiled

    @property
    def is_cuda_graph_compiled(self) -> bool:
        return self._is_cuda_graph_compiled

    @property
    def cuda_graph_executor(self) -> CudaGraphPoolExecutor:
        if self._cuda_graph_executor is None:
            raise RuntimeError("CUDA graph executor not compiled. Call compile_cuda_graphs() first.")

        return self._cuda_graph_executor

    @property
    def _warmup_inputs(self) -> list[StrTensorDict]:
        return []  # returns a list of inputs to warmup the forward method

    @property
    def _dynamo_options(self) -> dict[str, Any]:
        return {}

    def compile(self) -> Self:
        """
        Compile and warm up the model with TorchDynamo and the inputs, as defined by
        :py:meth:`.TorchDynamoMixin._warmup_inputs`. Returns the compiled version of the model.
        """
        from ypl.pytorch.model.base import YuppModel

        assert isinstance(self, YuppModel)

        self.eval()

        if self._is_dynamo_compiled:
            raise RuntimeError("Model already compiled")

        torch._dynamo.config.cache_size_limit = self._dynamo_cache_size_limit
        compiled_obj = torch.compile(self, **self._dynamo_options)
        compiled_obj._is_dynamo_compiled = True  # type: ignore[attr-defined]
        torch.set_float32_matmul_precision("high")  # to use tensor cores

        for i, input in enumerate(self._warmup_inputs):
            logging.info(f"Warming up model (iteration {i + 1}/{len(self._warmup_inputs)})")
            compiled_obj(input)  # warmup

        logging.info("Model warmed up")

        return compiled_obj  # type: ignore[return-value]
