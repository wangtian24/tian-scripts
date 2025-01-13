from ypl.backend.llm.routing.modules.base import RouterModule
from ypl.backend.llm.routing.router_state import RouterState


class Passthrough(RouterModule):
    def _select_models(self, state: RouterState) -> RouterState:
        return state

    async def _aselect_models(self, state: RouterState) -> RouterState:
        return state


class ConsoleDebugPrinter(RouterModule):
    def _select_models(self, state: RouterState) -> RouterState:
        print(state)
        return state

    async def _aselect_models(self, state: RouterState) -> RouterState:
        print(state)
        return state


class ModifierAnnotator(RouterModule):
    """Just adds the applicable modifiers to the state."""

    def __init__(self, modifiers: list[str]) -> None:
        self.modifiers = modifiers

    def _select_models(self, state: RouterState) -> RouterState:
        return state.emplaced(applicable_modifiers=self.modifiers)

    async def _aselect_models(self, state: RouterState) -> RouterState:
        return state.emplaced(applicable_modifiers=self.modifiers)
