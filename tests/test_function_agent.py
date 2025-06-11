
import uuid
import pytest
import logging

from agent_apps.model_analyzer.agent.ds_model_agents import PharmaModelAnalyzerAgent
from agent_apps.model_analyzer.agent.ds_model_kernels import ModelAnalyzerKernel
from agent_apps.model_analyzer.session.model_analyzer_session import ModelAnalyzerSession
from agent_apps.model_analyzer.tools.f1_score_tool import get_f1_score_tool
from agent_apps.model_analyzer.tools.plot_confusion_matrix_tool import plot_confusion_matrix_image_tool
from agent_apps.model_analyzer.tools.shap_feature_imp_tool import shap_feature_importance_data
from agent_apps.model_analyzer.tools.shap_insight_narrative_tool import shap_insight_narrative_tool
from agent_apps.model_analyzer.tools.shap_summary_plot_tool import shap_summary_plot_image_tool
from agent_apps.model_analyzer.tools.tool_names import ModelAnalyzerToolName
from app.cache.joblib_session_cache import JoblibSessionCache
from app.core.agent.savant_function_agent import SavantFunctionAgent
from app.core.agent.kernel.workflow_agent.function_agent import FunctionAgent
from app.core.utils.agent_result_utils import AgentResultUtils
from app.llms.llm_manager import LLMManager
from app.llms.mock_function_llm import MockFunctionCallingLLM
from app.storage.file_object_reader import FileObjectReader
from llama_index.core.workflow.context import Context
import matplotlib
from llama_index.core.agent.workflow.workflow_events import (
    AgentOutput
)

matplotlib.use("Agg")


@pytest.mark.asyncio
async def test_function_agent():
    llm = LLMManager.get_llm(model_name="40-mini")
    agent = FunctionAgent(
        tools=[shap_insight_narrative_tool, shap_summary_plot_image_tool],
        llm=llm,
        system_prompt="You are a helpful assistant for AI Model insights and explainability in pharma industry ",
    )
    response = await agent.run(user_msg="summarize insights for top 5 features for given session id 1")
    logging.info(f"response :{response}")


@pytest.mark.asyncio
async def test_function_agent_image():
    reader = FileObjectReader("tests/test_data/pateint_fullfillment_model.pkl")
    memory = JoblibSessionCache.get_instance()
    session = ModelAnalyzerSession(memory)
    session_id = session.create_session(reader)
    conversation_id = str(uuid.uuid4())
    # llm = LLMManager.get_llm(model_name="40-mini")
    llm = MockFunctionCallingLLM(max_tokens=128000)
    agent = PharmaModelAnalyzerAgent(
        tools=[shap_feature_importance_data, shap_summary_plot_image_tool, shap_insight_narrative_tool, get_f1_score_tool, plot_confusion_matrix_image_tool],
        llm=llm,
    )
    context = Context(workflow=agent)
    await context.set("session_id",session_id)
    await context.set("conversation_id",conversation_id)
    await context.set("model_analyzer_session",session)
    
    response:AgentOutput = await agent.run(user_msg=f"Generate Shap summary plot  ", ctx=context,
                               kernel_cls=ModelAnalyzerKernel)
    assert response is not None
    response_content = str(response)
    savant_agent_output = await AgentResultUtils.parse_result_output(context, response)
    agent_response_content = str(savant_agent_output)
    logging.info(f"response :{agent_response_content}")    
