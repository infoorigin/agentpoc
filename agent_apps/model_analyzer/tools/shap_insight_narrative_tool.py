
import numpy as np
from llama_index.core.tools import FunctionTool

from agent_apps.model_analyzer.session.model_analyzer_session import ModelAnalyzerSession
from agent_apps.model_analyzer.tools.tool_names import ModelAnalyzerToolName
from app.llms.llm_manager import LLMManager  # You can replace with your preferred LLM

def generate_shap_insight_narrative(
    session_id: str,
    feature_num: int
) -> str:
    """
    Generate business-friendly SHAP insights and narrative of the trained model from pharma domain for a given session.

    Args:
        session_id (str): The ID for the current model analysis session.
        feature_num (int): Number of features to include in the insights.

    Returns:
        str: A structured, human-readable narrative with insights and recommendations.
    """
    llm = LLMManager.get_llm(model_name="40-mini")
    analyzer_session = ModelAnalyzerSession.get_instance()
    shap_values = analyzer_session.get_shap_values(session_id)               
    feature_names = analyzer_session.get_features(session_id)

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:feature_num]

    feature_descriptions = []
    for i in top_indices:
        fname = feature_names[i]
        shap_col = shap_values[:, i]
        importance = mean_abs[i]

        # Direction based on % of SHAP values > 0
        positive_ratio = (shap_col > 0).mean()
        if positive_ratio > 0.75:
            direction = "↑ mostly increases fulfillment"
        elif positive_ratio < 0.25:
            direction = "↓ mostly decreases fulfillment"
        else:
            direction = "↔ mixed or neutral impact"

        feature_descriptions.append(
            f"- **{fname}**: importance = {importance:.3f}, trend = {direction}"
        )

    feature_text = "\n".join(feature_descriptions)

    prompt = f"""
        You are provided with SHAP values and feature names from a patient fulfillment prediction model in the pharma industry.

        The top features ranked by SHAP importance are:

        {feature_text}

        Each feature includes:
        - SHAP **importance** (how influential it is)
        - SHAP **trend** (direction of impact: ↑ increases fulfillment, ↓ decreases, ↔ mixed)

        ---

        🔍 **Your task**:

        Generate a concise, structured, and business-oriented insights summary using the information above.

        Follow this exact format:

        ---

        ### 🔍 Insights Summary on Patient Fulfillment Prediction

        #### 1. 🚀 Top Drivers of Fulfillment
        - Briefly explain what the top features indicate about patient behavior or care dynamics.
        - Use plain language with clear directionality.

        #### 2. 📊 Business Interpretation
        - For each top feature, explain **what it means in the real world**.
        - Translate features like `hcp_*`, `practice_size`, or `out_of_pocket_cost` into operational terms.
        - Link each feature to a potential barrier or enabler of patient success.

        #### 3. 🧭 Behavioral & Operational Patterns
        - Identify patterns across categories: e.g., demographic, provider behavior, access mechanism (hub), or financial strain.
        - Summarize what the data suggests about patient fulfillment pathways.

        #### 4. 🛠️ Actionable Recommendations
        - Recommend **strategies pharma teams can implement** to improve fulfillment.
        - Prioritize practical, high-leverage actions (e.g., provider training, patient onboarding, financial support programs).

        📝 **Tone**:
        - Professional, concise, and executive-ready.
        - Avoid generic AI speak. Be specific and grounded.
        - Emphasize clarity and usability of insights.
        """


    return llm.complete(prompt).text


shap_insight_narrative_tool = FunctionTool.from_defaults(
    async_fn=generate_shap_insight_narrative,
    name= ModelAnalyzerToolName.SHAP_INSIGHT_NARRATIVE,
)