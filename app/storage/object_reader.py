from abc import ABC, abstractmethod
from typing import BinaryIO

class ObjectReader(ABC):
    @abstractmethod
    def read(self) -> BinaryIO:
        """
        Returns a file-like binary object suitable for pickle.load().
        Should return an open stream (e.g., from open(file, 'rb') or BytesIO).
        """
        pass
