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
