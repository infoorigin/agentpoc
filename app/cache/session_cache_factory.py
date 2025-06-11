import os
from app.cache.session_cache_manager import SessionCacheManager
from app.cache.joblib_session_cache import JoblibSessionCache
from app.cache.redis_session_cache import RedisSessionCache
# from app.cache.s3_session_cache import S3SessionCache  # Optional: add later


class SessionCacheFactory:
    """
    Factory to instantiate a SessionCacheManager backend.

    Supports:
    - joblib: in-memory or file-based
    - redis: remote Redis server
    - (optional) s3: S3-based object cache
    """

    @staticmethod
    def get_cache_manager() -> SessionCacheManager:
        backend = os.getenv("SESSION_CACHE_BACKEND", "joblib").lower()

        if backend == "joblib":
            location_mode = os.getenv("JOBLIB_CACHE_LOCATION", "file").lower()
            if location_mode == "memory":
                return JoblibSessionCache.get_instance(location=None)
            else:
                path = os.getenv("JOBLIB_CACHE_DIR", "./cache_data")
                return JoblibSessionCache.get_instance(location=path)

        elif backend == "redis":
            return RedisSessionCache.get_instance()

        # Optional future support for S3
        # elif backend == "s3":
        #     return S3SessionCache.get_instance()

        else:
            raise ValueError(f"Unsupported SESSION_CACHE_BACKEND: '{backend}'")
