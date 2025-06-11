import base64
import io
from typing import Optional
from pydantic import BaseModel, Field
import shap
import matplotlib.pyplot as plt
import numpy as np
from llama_index.core.workflow.context import Context
from llama_index.core.tools import FunctionTool

from agent_apps.model_analyzer.session.model_analyzer_session import ModelAnalyzerSession
from agent_apps.model_analyzer.tools.tool_names import ModelAnalyzerToolName
from app.llms.llm_manager import LLMManager
from app.models.agent_models import ToolResultOutput  # You can replace with your preferred LLM



async def generate_shap_summary_plot_image(
    ctx: Context,
    feature_num: Optional[int] = None
) -> ToolResultOutput:
    """
    Generate a SHAP summary plot image as a base64-encoded string or all or top N features.

    This function retrieves the model analysis session using the provided session ID,
    generates a summary plot of the SHAP values (for all or top N features),
    and returns it as a base64-encoded PNG image


    Args:
        ctx (Context):A global object representing a context for a given workflow run.
        feature_num (int): Number of  features to include in the insights.
        
    Returns:
        str: A base64-encoded PNG image string representing the SHAP summary plot.

    Raises:
        ValueError: If the session cannot be found or required SHAP data is missing.
    """


    analyzer_session:ModelAnalyzerSession = await ctx.get("model_analyzer_session")
    session_id:str = await ctx.get("session_id")
    conversation_id:str = await ctx.get("conversation_id")
    shap_values = analyzer_session.get_shap_values(session_id)
    X = analyzer_session.get_X_test_transformed(session_id)

    if not analyzer_session or shap_values is None or X is None:
        raise ValueError(f"Missing analyzer_session or required SHAP data for session_id: {session_id}")

    if feature_num is not None and feature_num < X.shape[1]:
        mean_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_shap)[-feature_num:][::-1]
        shap_values = shap_values[:, top_indices]
        X = X.iloc[:, top_indices]
    
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    plt.close()
    buffer.seek(0)

    encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
    tool_result = ToolResultOutput(tool_name=ModelAnalyzerToolName.SHAP_SUMMARY_PLOT,
        conversation_id=conversation_id, content_type="BASE64IMAGE", content=encoded_image, session_id=session_id)
    return tool_result

async def generate_shap_summary_plot_image_text(response :  ToolResultOutput | None):
    if response is not None:
        return f"Shap Summary image genarated for session_id {response.session_id}"
        
    return f"Shap Summary image can not be generated for session_id {response.session_id} "

shap_summary_plot_image_tool = FunctionTool.from_defaults(
    async_fn=generate_shap_summary_plot_image,
    name= ModelAnalyzerToolName.SHAP_SUMMARY_PLOT,
    async_callback=generate_shap_summary_plot_image_text

)