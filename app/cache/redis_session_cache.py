import redis
import pickle
from threading import Lock
from typing import Optional

from app.cache.session_cache_manager import SessionCacheManager


class RedisSessionCache(SessionCacheManager):
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
        self.redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=False  # binary mode
        )
        self.default_ttl = int(os.getenv("REDIS_DEFAULT_TTL", 3600))  # 1 hour default

    def _get_key(self, id: str, datatype: str) -> str:
        return f"{datatype}:{id}"

    def save(self, id: str, datatype: str, data: object, ttl: Optional[int] = None) -> None:
        key = self._get_key(id, datatype)
        value = pickle.dumps(data)
        self.redis.setex(key, ttl or self.default_ttl, value)

    def load(self, id: str, datatype: str) -> Optional[object]:
        key = self._get_key(id, datatype)
        value = self.redis.get(key)
        return pickle.loads(value) if value is not None else None

    def exists(self, id: str, datatype: str) -> bool:
        key = self._get_key(id, datatype)
        return self.redis.exists(key) == 1

    def delete(self, id: str, datatype: str) -> None:
        key = self._get_key(id, datatype)
        self.redis.delete(key)
