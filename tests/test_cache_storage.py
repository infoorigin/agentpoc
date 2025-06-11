


from agent_apps.model_analyzer.session.model_analyzer_session import ModelAnalyzerSession
from app.cache.joblib_session_cache import JoblibSessionCache
from app.storage.file_object_reader import FileObjectReader


def test_model_analyzer_session():
    reader = FileObjectReader("tests/test_data/pateint_fullfillment_model.pkl")
    memory = JoblibSessionCache.get_instance()
    session = ModelAnalyzerSession(memory)
    session_id = session.create_session(reader)
    model = session.get_model(session_id)

    assert model is not None
