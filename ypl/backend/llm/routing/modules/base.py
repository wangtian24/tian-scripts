import logging
from abc import ABC, abstractmethod

import numpy as np
from ypl.backend.llm.routing.router_state import RouterState
from ypl.utils import RNGMixin


class RouterModule(ABC):
    """
    Abstract base class for a module in the routing pipeline. Routing is defined as passing a state through an
    arithmetic circuit of modules, which may include selection, filtering, mapping, branching, and sequential
    and parallel composition, akin to MapReduce or a Unix pipeline. A router is composed of a sequence of
    arithmetic operations on RouterModule, e.g.,

    .. code-block::python

        router = CostModelProposer() | TopK(3)
        state = router.select_models(state=RouterState(all_models=set(FRONTEND_MODELS)))
        print(state.get_selected_models())

    would route to the top 3 cheapest models out of the available front-end models, passing the state
    sequentially through CostModelProposer to TopK. These three arithmetic operations are supported:

    .. code-block::python

        # Sequential composition; the state is passed sequentially through each module
        router = CostModelProposer() | EloProposer(ranker)

        # Parallel composition; the state is split across two modules, which operate in parallel.
        # The results of each module are merged together (see :py:meth:`.RouterState.__add__`).
        router = CostModelProposer() & EloProposer(ranker)

        # Exclusive drawing; the state is randomly routed to one of the two modules, with a probability
        # given by the `with_probs` method.
        router = CostModelProposer() ^ EloProposer(ranker)

    These modules can be chained and combined arbitrarily, e.g.,

    .. code-block::python

        router = (
            (((EloProposer(ranker) & CostModelProposer()) | TopK(3))
            ^ ConfidenceIntervalWidthModelProposer(ranker)).with_probs(0.75, 0.25)
        ) | RoutingDecisionLogger()

    See Also:
    - :py:meth:`.RouterSequentialChain`, :py:meth:`.RouterParallelChain`, :py:meth:`.RouterExclusiveChain`
    - :py:meth:`.RouterState.__add__`, :py:meth:`.RouterState.multiply_scores`
    - :py:meth:`.RoutingDecisionLogger`
    """

    _multiplier: float | None = None
    _always_include: bool | None = None
    _offset: float | None = None

    def select_models(self, state: RouterState | None = None) -> RouterState:
        response = self._select_models(state or RouterState())

        if self._multiplier is not None:
            response.multiply_scores(self._multiplier)

        if self._offset is not None:
            response.offset_scores(self._offset)

        if self._always_include is not None:
            response.always_include = self._always_include

        return response

    async def aselect_models(self, state: RouterState | None = None) -> RouterState:
        response = await self._aselect_models(state or RouterState())

        if self._multiplier is not None:
            response.multiply_scores(self._multiplier)

        if self._offset is not None:
            response.offset_scores(self._offset)

        if self._always_include is not None:
            response.always_include = self._always_include

        return response

    def with_flags(
        self,
        *,
        multiplier: float | None = None,
        always_include: bool | None = None,
        offset: float | None = None,
    ) -> "RouterModule":
        if multiplier is not None:
            self._multiplier = multiplier

        if offset is not None:
            self._offset = offset

        if always_include is not None:
            self._always_include = always_include

        return self

    def __or__(self, other: "RouterModule") -> "RouterModule":
        return RouterSequentialChain(self, other)

    def __and__(self, other: "RouterModule") -> "RouterModule":
        return RouterParallelChain(self, other)

    def __xor__(self, other: "RouterModule") -> "RouterExclusiveChain":
        return RouterExclusiveChain(self, other)

    @abstractmethod
    def _select_models(self, state: RouterState) -> RouterState:
        raise NotImplementedError

    async def _aselect_models(self, state: RouterState) -> RouterState:
        return self._select_models(state)


class RouterSequentialChain(RouterModule):
    def __init__(self, *args: RouterModule) -> None:
        """
        Represents a sequential chain of routing modules, i.e., the state is passed sequentially through each module.
        See :py:class:`.RouterModule` for more information.

        Args:
            *args: The modules to pass the state through sequentially.
        """
        self.router_modules: list[RouterModule] = list(args)

    def __or__(self, other: "RouterModule") -> "RouterModule":
        self.router_modules.append(other)
        return self

    def _select_models(self, state: RouterState) -> RouterState:
        for router_module in self.router_modules:
            state = router_module.select_models(state=state)

        return state

    async def _aselect_models(self, state: RouterState) -> RouterState:
        for router_module in self.router_modules:
            state = await router_module.aselect_models(state=state)

        return state


class RouterParallelChain(RouterModule):
    def __init__(self, *args: RouterModule) -> None:
        """
        Represents a parallel chain of routing modules, i.e., the state is split across all modules.
        See :py:class:`.RouterModule` for more information.

        Args:
            *args: The modules to pass the state through in parallel.
        """
        self.router_modules: list[RouterModule] = list(args)

    def __and__(self, other: "RouterModule") -> "RouterModule":
        self.router_modules.append(other)
        return self

    def _select_models(self, state: RouterState) -> RouterState:
        responses = []

        for router_module in self.router_modules:
            router_response = router_module.select_models(state=state.deepcopy())
            responses.append(router_response)

        for response in responses:
            state += response

        return state

    async def _aselect_models(self, state: RouterState) -> RouterState:
        responses = []

        for router_module in self.router_modules:
            router_response = await router_module.aselect_models(state=state.deepcopy())
            responses.append(router_response)

        for response in responses:
            state += response

        return state


class RouterExclusiveChain(RNGMixin, RouterModule):
    def __init__(self, *args: RouterModule) -> None:
        """
        Represents an exclusive chain of routing modules, i.e., the state is randomly routed to one of the modules.
        See :py:class:`.RouterModule` for more information.

        Args:
            *args: The modules to randomly route the state to.
        """
        self.router_modules: list[RouterModule] = list(args)
        self.random_probabilities: list[float] = []

    def __xor__(self, other: "RouterModule") -> "RouterExclusiveChain":
        self.router_modules.append(other)
        return self

    def with_probs(self, *probabilities: float) -> "RouterExclusiveChain":
        """
        Set the probabilities of routing to each module in the same order that it was constructed.

        Args:
            *probabilities: The probabilities of routing to each module.
        """
        p = np.array(probabilities)
        p = p / p.sum()
        self.random_probabilities = p.tolist()

        return self

    def _choose_module(self) -> RouterModule:
        if len(self.random_probabilities) != len(self.router_modules):
            logging.warning(
                "Random probabilities not set for RouterExclusiveChain; using default of 1/len(router_modules)"
            )
            probs = np.full(len(self.router_modules), 1 / len(self.router_modules))
        else:
            probs = np.array(self.random_probabilities)

        return self.get_rng().choice(  # type: ignore[no-any-return]
            np.array(self.router_modules, dtype=object), replace=False, p=probs
        )

    def _select_models(self, state: RouterState) -> RouterState:
        chosen_module = self._choose_module()
        return chosen_module.select_models(state=state)

    async def _aselect_models(self, state: RouterState) -> RouterState:
        chosen_module = self._choose_module()
        return await chosen_module.aselect_models(state=state)
