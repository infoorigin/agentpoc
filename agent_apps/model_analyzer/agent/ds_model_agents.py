from typing import Callable, List, Optional, Union
from agent_apps.model_analyzer.tools.shap_insight_narrative_tool import generate_shap_insight_narrative
from agent_apps.model_analyzer.tools.shap_summary_plot_tool import generate_shap_summary_plot_image
from app.core.agent.savant_function_agent import SavantFunctionAgent
from llama_index.core.tools import BaseTool
from llama_index.core.llms import LLM
from llama_index.core.objects import ObjectRetriever

from app.llms.llm_manager import LLMManager


PHARMA_MODEL_ANALYZER_ROLE = "AI Model Insights Assistant"

PHARMA_MODEL_ANALYZER_BACKGROUND = (
    "You support pharma commercial teams by reviewing AI/ML outputs on prescription data, "
    "patient access, and fulfillment behavior."
)

PHARMA_MODEL_ANALYZER_GOAL = (
    "to convert model results into actionable insights for brand strategy, adherence programs, "
    "and market execution."
)

class PharmaModelAnalyzerAgent(SavantFunctionAgent):
    """
    An agent tailored for AI model insight generation and explainability in the pharma commercial domain.
    Uses default role, background, and goal, but allows overriding tools or LLM.

    Example:
        >>> agent = PharmaModelAnalyzerAgent()
        >>> agent_custom = PharmaModelAnalyzerAgent(
        ...     goal="Support field teams with actionable performance insights."
        ... )
    """

    def __init__(
        self,
        name: str = "PharmaModelAnalyzerAgent",
        role: str = PHARMA_MODEL_ANALYZER_ROLE,
        background: str = PHARMA_MODEL_ANALYZER_BACKGROUND,
        goal: Optional[str] = PHARMA_MODEL_ANALYZER_GOAL,
        tools: Optional[List[Union[BaseTool, Callable]]] = None,
        llm: Optional[LLM] = None,
        tool_retriever: Optional[ObjectRetriever] = None,
        can_handoff_to: Optional[List[str]] = None,
        language: str = "en",
    ):
        super().__init__(
            role=role,
            background=background,
            goal=goal,
            name=name,
            tools=tools or [generate_shap_insight_narrative, generate_shap_summary_plot_image],
            tool_retriever=tool_retriever,
            can_handoff_to=can_handoff_to,
            llm=llm or LLMManager.get_llm(model_name="40-mini"),
            language=language,
        )