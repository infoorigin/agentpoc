import logging
import uuid

import pytest
from agent_apps.model_analyzer.model.patient_fulfillment_modelbuilder import PatientFulfillmentModelBuilder
from agent_apps.model_analyzer.session.model_analyzer_session import ModelAnalyzerSession
from agent_apps.model_analyzer.tools.shap_feature_imp_tool import shap_feature_importance_data
from agent_apps.model_analyzer.tools.shap_insight_narrative_tool import generate_shap_insight_narrative
from agent_apps.model_analyzer.tools.shap_summary_plot_tool import generate_shap_summary_plot_image
from agent_apps.model_analyzer.utils.shap_plot_generator import SHAPPlotGenerator
from app.cache.joblib_session_cache import JoblibSessionCache
from app.core.agent.savant_function_agent import SavantFunctionAgent
from app.llms.mock_function_llm import MockFunctionCallingLLM
from app.storage.file_object_reader import FileObjectReader
from app.tools.ds_models.model_loader import ModelLoader
from llama_index.core.workflow.context import Context
from app.core.agent.kernel.workflow_agent.multi_agent_workflow import AgentWorkflow

import matplotlib
matplotlib.use('Agg')

def test_create_patient_model_pickle():
    builder = PatientFulfillmentModelBuilder(n_samples=1000)
    # Build everything and save
    builder.build_and_save_model(output_file='data/pateint_fullfillment_model.pkl')
    logging.info("model pickel file created")

def test_pateint_generate_shap_plots():
    loader = ModelLoader('data/pateint_fullfillment_model.pkl')
    loader.load_pickle()
    model, feature_names, X_test, y_test = loader.get_model_data()
    # 2. Generate SHAP plots
    shap_generator = SHAPPlotGenerator(model=model, X_test=X_test, output_dir='shap_outputs')
    shap_generator.full_generate_all_plots(top_n_features=5)
    logging.info("shap plots generated")

@pytest.mark.asyncio
async def test_shap_importance_data():
    reader = FileObjectReader("tests/test_data/pateint_fullfillment_model.pkl")
    memory = JoblibSessionCache.get_instance()
    session = ModelAnalyzerSession(memory)
    session_id = session.create_session(reader)
    conversation_id = str(uuid.uuid4())
    llm = MockFunctionCallingLLM(max_tokens=128000)
    mock_agent = SavantFunctionAgent(
                llm=llm,
                role = "role",
                background= "background",
                goal= "goal",
                tools=[]
            )
    context = Context(workflow=mock_agent)
    await context.set("session_id",session_id)
    await context.set("conversation_id",conversation_id)
    await context.set("model_analyzer_session",session)
    output = await shap_feature_importance_data(ctx=context, feature_num=5)
    assert output is not None


def test_shap_summary_image():
    reader = FileObjectReader("tests/test_data/pateint_fullfillment_model.pkl")
    memory = JoblibSessionCache.get_instance()
    session = ModelAnalyzerSession(memory)
    session_id = session.create_session(reader)
    base_img = generate_shap_summary_plot_image(session_id, 20)
    assert base_img is not None


def test_call_generate_narative():
    reader = FileObjectReader("tests/test_data/pateint_fullfillment_model.pkl")
    memory = JoblibSessionCache.get_instance()
    session = ModelAnalyzerSession(memory)
    session_id = session.create_session(reader)
    feature_names = session.get_features(session_id)
    assert feature_names is not None
    
    narrative = generate_shap_insight_narrative(
        session_id=session_id,
        feature_num=5  
    )

    logging.info(f"narrative :{narrative}")

