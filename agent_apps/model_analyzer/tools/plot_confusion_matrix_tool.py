import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from llama_index.core.workflow.context import Context
from io import BytesIO
import base64
from llama_index.core.tools import FunctionTool


from agent_apps.model_analyzer.session.model_analyzer_session import ModelAnalyzerSession

from agent_apps.model_analyzer.tools.tool_names import ModelAnalyzerToolName
from app.models.agent_models import ToolResultOutput

async def plot_confusion_matrix_image(ctx: Context) -> ToolResultOutput:
    """
    Generate a base64-encoded PNG image of the model's confusion matrix.

    This tool is useful for visually interpreting model performance by comparing 
    predicted vs actual labels. The output can be embedded into dashboards or 
    consumed by downstream GenAI agents.

    Returns:
        str: Base64-encoded string of the confusion matrix image in PNG format.
    
    Example:
        >>> base64_img = plot_confusion_matrix_base64()
        >>> # Use in HTML: <img src="data:image/png;base64,{base64_img}" />
    
    Tool Tags:
        - explainability
    """

    analyzer_session:ModelAnalyzerSession = await ctx.get("model_analyzer_session")
    session_id:str = await ctx.get("session_id")
    conversation_id:str = await ctx.get("conversation_id")
    X_test = analyzer_session.get_X_test(session_id)
    y_test = analyzer_session.get_y_test(session_id)
    model = analyzer_session.get_model(session_id) 
    
    if model is None or X_test is None or y_test is None:
        raise ValueError("Model and test data must be trained/preprocessed first.")

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=["Not Fulfilled", "Fulfilled"],
                yticklabels=["Not Fulfilled", "Fulfilled"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    plt.close()
    buffer.seek(0)

    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    tool_result = ToolResultOutput(tool_name=ModelAnalyzerToolName.SHAP_SUMMARY_PLOT,
        conversation_id=conversation_id, content_type="BASE64IMAGE", content=img_base64, session_id=session_id)
    return tool_result


async def plot_confusion_matrix_image_text(response :  ToolResultOutput):
    if response is not None:
        return f"Confusion Matrix image genarated for session_id {response.session_id}"
        
    return f"hap Summary image can not be generated for session_id {response.session_id} "

plot_confusion_matrix_image_tool = FunctionTool.from_defaults(
    async_fn=plot_confusion_matrix_image,
    name= ModelAnalyzerToolName.CONFUSION_MATRIX_PLOT,
    async_callback=plot_confusion_matrix_image_text

)