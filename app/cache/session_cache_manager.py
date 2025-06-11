from abc import ABC, abstractmethod

class SessionCacheManager(ABC):
    @abstractmethod
    def save(self, id: str, datatype: str, data: object) -> None:
        pass

    @abstractmethod
    def load(self, id: str, datatype: str) -> object:
        pass

    @abstractmethod
    def exists(self, id: str, datatype: str) -> bool:
        pass

    @abstractmethod
    def delete(self, id: str, datatype: str) -> None:
        pass
