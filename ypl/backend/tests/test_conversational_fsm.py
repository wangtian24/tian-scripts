from collections import Counter

import pytest

from ypl.backend.llm.conversational.base import StatefulAgent, register_state


class DummyAgent(StatefulAgent[[str], str]):
    @register_state(entrypoint=True, weight=1)
    def dummy1(self, input: str) -> str:
        self.move_to(self.fn1)
        return "dummy1"

    @register_state(entrypoint=True, weight=2)
    def dummy2(self, input: str) -> str:
        self.move_to(self.fn1)
        return "dummy2"

    @register_state()
    def fn1(self, input: str) -> str:
        self.move_to(self.fn2)
        return "fn1"

    @register_state()
    def fn2(self, input: str) -> str:
        self.end()
        return "fn2"


@pytest.mark.parametrize("agent", [DummyAgent()])
def test_init(agent: DummyAgent) -> None:
    assert agent("input") in ("dummy1", "dummy2")
    assert agent.current_state == "fn1"


@pytest.mark.parametrize("agent", [DummyAgent()])
def test_run(agent: DummyAgent) -> None:
    while agent.running:
        agent("input")

    assert agent.current_state == agent.END_STATE


def test_entrypoint_distn() -> None:
    c: Counter[str] = Counter()

    for seed in range(100):
        agent = DummyAgent()
        agent.with_seed(seed)
        init_val = agent("input")
        c[init_val] += 1

    assert c["dummy1"] == pytest.approx(33, abs=10)
    assert c["dummy2"] == pytest.approx(66, abs=20)


def test_correct_route() -> None:
    agent = DummyAgent()
    agent.with_seed(0)
    assert agent("input") in ("dummy1", "dummy2")
    assert agent("input") == "fn1"
    assert agent("input") == "fn2"

    with pytest.raises(ValueError):
        agent("input")
