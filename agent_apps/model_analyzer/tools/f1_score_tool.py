
from sklearn.metrics import f1_score
from llama_index.core.workflow.context import Context

from llama_index.core.tools import FunctionTool
from agent_apps.model_analyzer.session.model_analyzer_session import ModelAnalyzerSession
from agent_apps.model_analyzer.tools.tool_names import ModelAnalyzerToolName

async def get_f1_score(ctx: Context) -> str:
    """
    Compute and return the F1 score of the trained patient fulfillment model.

    This tool evaluates the classification performance of the model by calculating 
    the F1 score using predictions on the test set. Useful for summarizing model 
    effectiveness in handling both precision and recall.

    Returns:
        float: F1 score of the model on the test data.

    Example:
        >>> score = get_f1_score()
        >>> print(f"F1 Score: {score:.2f}")

    """

    analyzer_session:ModelAnalyzerSession = await ctx.get("model_analyzer_session")
    session_id:str = await ctx.get("session_id")
    X_test = analyzer_session.get_X_test(session_id)
    y_test = analyzer_session.get_y_test(session_id)
    model = analyzer_session.get_model(session_id) 
    y_pred = model.predict(X_test)
    return round(f1_score(y_test, y_pred), 4)
    
get_f1_score_tool = FunctionTool.from_defaults(
    async_fn=get_f1_score,
    name= ModelAnalyzerToolName.F1_SCORE,
)