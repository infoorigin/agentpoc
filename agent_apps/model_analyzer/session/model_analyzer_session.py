import pickle
from threading import Lock
import uuid

from agent_apps.model_analyzer.utils.shap_plot_generator import SHAPPlotGenerator
from app.cache.session_cache_manager import SessionCacheManager
from app.storage.object_reader import ObjectReader

class ModelAnalyzerSession:
    _instance = None
    _lock = Lock()

    def __new__(cls, session_cache:SessionCacheManager=None):
        with cls._lock:
            if cls._instance is None:
                if session_cache is None:
                    raise ValueError("First call to ModelAnalyzerSession must provide a memory_manager.")
                cls._instance = super().__new__(cls)
                cls._instance._init(session_cache)
        return cls._instance

    def _init(self, session_cache:SessionCacheManager):
        self.session_cache = session_cache

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise RuntimeError("ModelAnalyzerSession is not initialized. Call with memory_manager first.")
        return cls._instance

    def create_session(self, reader: ObjectReader, session_id: str = None) -> str:
        """
        Creates a session by reading a model via the ObjectReader, computing SHAP values,
        and caching the results.

        Args:
            reader (ObjectReader): Source to load model pickle.
            session_id (str, optional): Unique session ID. If None, one will be generated.

        Returns:
            str: The session ID used.
        """
        session_id = session_id or str(uuid.uuid4())

        if self.session_cache.exists(session_id, "shap"):
            return session_id  # Session already cached

        with reader.read() as f:
            model_data = pickle.load(f)

        model = model_data["model"]
        X_test = model_data["X_test"]
        y_test = model_data["y_test"]
        X_test_transformed = model_data.get("X_test_transformed", X_test)

        shap_generator = SHAPPlotGenerator(model=model, X_test=X_test_transformed, output_dir="shap_outputs")
        shap_generator.calculate_shap_values()

        self.session_cache.save(session_id, "model", model)
        self.session_cache.save(session_id, "features", X_test_transformed.columns.tolist())
        self.session_cache.save(session_id, "shap", shap_generator.shap_values)
        self.session_cache.save(session_id, "X_test", X_test)
        self.session_cache.save(session_id, "y_test", y_test)
        self.session_cache.save(session_id, "X_test_transformed", X_test_transformed)

        return session_id

    def get_y_test(self, session_id: str):
        return self.session_cache.load(session_id, "y_test")
    
    def get_X_test(self, session_id: str):
        return self.session_cache.load(session_id, "X_test")

    def get_X_test_transformed(self, session_id: str):
        return self.session_cache.load(session_id, "X_test_transformed")
    
    def get_model(self, session_id: str):
        return self.session_cache.load(session_id, "model")

    def get_features(self, session_id: str):
        return self.session_cache.load(session_id, "features")

    def get_shap_values(self, session_id: str):
        return self.session_cache.load(session_id, "shap")
