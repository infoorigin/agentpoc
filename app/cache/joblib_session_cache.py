import os
import joblib
from threading import Lock

from app.cache.session_cache_manager import SessionCacheManager


class JoblibSessionCache(SessionCacheManager):
    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._initialize()
                    cls._instance = obj
        return cls._instance

    def _initialize(self):
        # Load from ENV or use default
        self.cache_dir = os.getenv("JOBLIB_CACHE_DIR", "./cache_data")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_path(self, id: str, datatype: str) -> str:
        return os.path.join(self.cache_dir, f"{id}_{datatype}.pkl")

    def save(self, id: str, datatype: str, data: object) -> None:
        joblib.dump(data, self._get_path(id, datatype), compress=("zlib", 3))

    def load(self, id: str, datatype: str) -> object:
        return joblib.load(self._get_path(id, datatype))

    def exists(self, id: str, datatype: str) -> bool:
        return os.path.exists(self._get_path(id, datatype))

    def delete(self, id: str, datatype: str) -> None:
        path = self._get_path(id, datatype)
        if os.path.exists(path):
            os.remove(path)
