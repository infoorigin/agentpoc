from typing import List, Dict, Any, Optional
import uuid
import pandas as pd
from llama_index.core.workflow.context import Context

from agent_apps.model_analyzer.session.model_analyzer_session import ModelAnalyzerSession
from agent_apps.model_analyzer.tools.tool_names import ModelAnalyzerToolName
from app.models.agent_models import ToolResultOutput
from llama_index.core.tools import FunctionTool


async def shap_feature_importance_data(ctx: Context,
    feature_num: Optional[int] = 10) -> ToolResultOutput:
    """
    Generate SHAP-based  feature importance values for the top N most important features

    Args:
        ctx (Context):A global object representing a context for a given workflow run.
        feature_num (int): Number of  features to include in the output (default = 10).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries with 'name' and 'value' keys representing
                              feature names and their SHAP-based importance scores.

    Example:
        >>> SHAPFeatureImportanceTool("patient_fulfillment_model.pkl", top_n=5)
        [
            {"name": "out_of_pocket_cost", "value": 0.0732},
            {"name": "hcp_biologics_experience", "value": 0.0611},
            {"name": "payer_plan_CVS Caremark", "value": 0.0456},
            ...
        ]
    """
    analyzer_session:ModelAnalyzerSession = await ctx.get("model_analyzer_session")
    session_id:str = await ctx.get("session_id")
    conversation_id:str = await ctx.get("conversation_id")
    shap_values = analyzer_session.get_shap_values(session_id)
    feature_names = analyzer_session.get_features(session_id)

    # Compute global feature importance
    shap_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": abs(shap_values[1]).mean(axis=0)
    }).sort_values(by="importance", ascending=False).head(feature_num)
    data  =  [
        {"name": row["feature"], "value": row["importance"]}
        for _, row in shap_importance_df.iterrows()
    ]
    tool_result = ToolResultOutput(tool_name=ModelAnalyzerToolName.SHAP_FEATURE_IMPORTANCE,
        conversation_id=conversation_id, content_type="JSON", content=data, session_id=session_id)
    # Return in JSON-ready format
    return tool_result

async def generate_shap_feature_importance_text(response :  ToolResultOutput):
    if response is not None:
        return f"Shap Feature data created  for session_id {response.session_id}"
        
    return f"Shap Feature data  can not be created for session_id {response.session_id} "

shap_feature_importance_tool = FunctionTool.from_defaults(
    async_fn=shap_feature_importance_data,
    name= ModelAnalyzerToolName.SHAP_FEATURE_IMPORTANCE,
    async_callback=generate_shap_feature_importance_text

)