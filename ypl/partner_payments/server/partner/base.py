from abc import ABC, abstractmethod


class BasePartnerClient(ABC):
    def __init__(self) -> None:
        self.config: dict | None = None

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        pass
