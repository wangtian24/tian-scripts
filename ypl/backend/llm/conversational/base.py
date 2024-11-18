from collections.abc import Callable, Coroutine
from typing import Any, Concatenate, Generic, ParamSpec, TypedDict, TypeVar

import numpy as np

from ypl.utils import RNGMixin

InputType = ParamSpec("InputType")
OutputType = TypeVar("OutputType")


def register_state(
    *,
    name: str = "",
    entrypoint: bool = False,
    weight: float = 1.0,
) -> Callable[[Callable[InputType, OutputType]], "StatelikeFunction[InputType, OutputType]"]:
    def decorator(fn: Callable[InputType, OutputType]) -> "StatelikeFunction[InputType, OutputType]":
        nonlocal name
        name = "__entrypoint__" if entrypoint else name or fn.__name__
        return StatelikeFunction(fn=fn, name=name, entrypoint=entrypoint, weight=weight)

    return decorator


class StatelikeFunction(Generic[InputType, OutputType]):
    def __init__(
        self,
        *,
        fn: Callable[InputType, OutputType],
        name: str,
        entrypoint: bool = False,
        weight: float = 1.0,
        bound_self: Any | None = None,
    ) -> None:
        self.fn = fn
        self.state_name = name
        self.entrypoint = entrypoint
        self.weight = weight
        self.bound_self = bound_self

    def __call__(self, *args: InputType.args, **kwargs: InputType.kwargs) -> OutputType:
        if self.bound_self is None:
            return self.fn(*args, **kwargs)
        else:
            return self.fn(self.bound_self, *args, **kwargs)

    def __get__(self, instance: Any, owner: Any) -> "StatelikeFunction[InputType, OutputType]":
        return StatelikeFunction(
            fn=self.fn,
            name=self.state_name,
            entrypoint=self.entrypoint,
            weight=self.weight,
            bound_self=instance,
        )


class AgentState(TypedDict):
    __current_state__: str


class StatefulAgent(RNGMixin, Generic[InputType, OutputType]):
    """
    A stateful conversational agent composed of statelike functions, each representing a node in a finite-state machine
    for building conversational workflows. Compared with existing codebases such as LangChain's LangGraph and AutoGen,
    this is much more lightweight and framework agnostic. The primary motivation is to build something that just works.

    Conversational agents should subclass this and implement registered functions using the :py:func:`.register_state`
    decorator. Transitions between nodes (registered functions) must be executed by calling `self.move_to(self.func)`.
    Each agent can be saved and loaded using `state_dict()` and `load_state_dict()`. Entry points are marked using
    the `entrypoint` argument to :py:func:`.register_state`. If there are multiple entry points, the agent will randomly
    choose one to start with, weighted by the `weight` argument. Likewise, if there are multiple functions with the same
    name, the agent will randomly choose one to execute, weighted by the `weight` argument.

    As a complete example,

    .. code-block:: python

        # This means we take one string input for all statelike functions and return a string output, similar to
        # the Callable[[ParamTypes...], ReturnType] syntax
        class WeatherAgent(StatefulAgent[[str], str]):
            @register_state(name="greetings", entrypoint=True)
            def generate_greetings(self, input: str | None = None) -> str:
                response = self.llm.generate("Generate a greeting and ask for the user's city")
                self.move_to(self.lookup_city)  # lookup_city will execute next

                return response

            @register_state()  # default name is the function name
            def lookup_city(self, user_input: str | None = None) -> str:
                response = self.llm_with_tool_use.generate(user_input)

                if response is None:
                    self.move_to(self.generate_city_not_found)  # move to generate_city
                    return self.generate_city_not_found(user_input)  # and execute
                else:
                    self.move_to(self.generate_goodbye)

                return response

            @register_state(name="generate_city_not_found")
            def generate_city_not_found(self, user_input: str | None = None) -> str:
                self.move_to(self.lookup_city)
                return "I couldn't find the city you're looking for. Please try again."

            @register_state(name="generate_goodbye")
            def generate_goodbye(self, user_input: str | None = None) -> str:
                self.end()  # end the agent
                return "Goodbye!"

        agent = WeatherAgent()

        while agent.running:
            user_input = input()
            output = agent(user_input)
            print(output)
    """

    NOT_BEGUN_STATE = "__not_begun__"
    END_STATE = "__end__"

    def __init__(self) -> None:
        self.state: AgentState = dict(__current_state__=self.NOT_BEGUN_STATE)

    @property
    def running(self) -> bool:
        return self.current_state != self.END_STATE

    @property
    def current_state(self) -> str:
        return self.state["__current_state__"]

    @current_state.setter
    def current_state(self, value: str) -> None:
        self.state["__current_state__"] = value

    def move_to(
        self,
        state: StatelikeFunction[Concatenate[Any, InputType], OutputType]
        | StatelikeFunction[Concatenate[Any, InputType], Coroutine[Any, Any, OutputType]],
    ) -> None:
        if not hasattr(state, "state_name"):
            raise ValueError("Need to register the function using @register_state")

        self.current_state = state.state_name

    def end(self) -> None:
        self.current_state = self.END_STATE

    def state_dict(self) -> AgentState:
        return self.state

    def load_state_dict(self, state_dict: AgentState) -> None:
        self.state = state_dict

    def get_begin_state(self) -> StatelikeFunction[InputType, OutputType]:
        entrypoints = []
        weights = []

        for fn_name in dir(self):
            fn = getattr(self, fn_name)

            if isinstance(fn, StatelikeFunction) and fn.entrypoint:
                entrypoints.append(fn)
                weights.append(fn.weight)

        if not entrypoints:
            raise ValueError("Entry point not found")

        p = np.array(weights) / sum(weights)

        return self.get_rng().choice(entrypoints, p=p)  # type: ignore

    def find_current_state(self) -> StatelikeFunction[InputType, OutputType]:
        match self.current_state:
            case self.NOT_BEGUN_STATE:
                return self.get_begin_state()
            case self.END_STATE:
                raise ValueError("Agent has ended already.")

        fns = []
        weights = []

        for fn_name in dir(self):
            fn = getattr(self, fn_name)

            if isinstance(fn, StatelikeFunction) and fn.state_name == self.current_state:
                fns.append(fn)
                weights.append(fn.weight)

        if not fns:
            raise ValueError("Current state not found")

        p = np.array(weights) / sum(weights)

        return self.get_rng().choice(fns, p=p)  # type: ignore

    def __call__(self, *args: InputType.args, **kwargs: InputType.kwargs) -> OutputType:
        return self.find_current_state()(*args, **kwargs)

    async def acall(self, *args: InputType.args, **kwargs: InputType.kwargs) -> OutputType:
        return await self.find_current_state()(*args, **kwargs)  # type: ignore
