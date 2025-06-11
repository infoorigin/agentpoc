from typing import List, Optional, Union, Callable
from llama_index.core.llms import LLM
from llama_index.core.tools import BaseTool
from llama_index.core.objects import ObjectRetriever
from llama_index.core.settings import Settings
from pydantic import PrivateAttr

from app.core.prompts.prompt_template_registry import PromptTemplateRegistry



class SavantAgentPersonaMixin:
    """
    Mixin that provides persona-based initialization with multilingual prompt templates.
    Uses 'agent' category templates from the PromptTemplateRegistry.

    Args:
        role (str): The agent's role.
        background (str): A short background description for the agent.
        goal (Optional[str]): Personal goal. If not provided, goal_block is omitted.
        name (str): Agent name.
        tools (List): Tools this agent can use.
        tool_retriever: Optional retriever.
        can_handoff_to: Optional list of handoff agent names.
        llm (LLM): Optional LLM instance.
        language (str): Language code (default = 'en'). Must match a registered template.

    Example (with built-in template):
        >>> agent = MyPersonaAgent(
        ...     role="Clinical Advisor",
        ...     background="You support late-phase oncology trials.",
        ...     goal="Ensure protocol compliance.",
        ...     language="en"
        ... )

    Example (register custom language first):
        >>> PromptTemplateRegistry.register_template("agent", "it", {
        ...     "system_prompt": "Sei {role}. {background}{goal_block}",
        ...     "goal_block": "\\nIl tuo obiettivo personale è: {goal}",
        ...     "description": "Un agente persona che agisce come {role}."
        ... })
        >>> agent = MyPersonaAgent(
        ...     role="Ricercatore",
        ...     background="Guida studi clinici in oncologia.",
        ...     goal="Identificare lacune nei dati.",
        ...     language="it"
        ... )
    """
    _role: str = PrivateAttr()
    _background: str = PrivateAttr()
    _goal: Optional[str] = PrivateAttr()
    
    def __init__(
        self,
        role: str,
        background: str,
        goal: Optional[str] = None,
        name: str = "PersonaAgent",
        tools: Optional[List[Union[BaseTool, Callable]]] = None,
        tool_retriever: Optional[ObjectRetriever] = None,
        can_handoff_to: Optional[List[str]] = None,
        llm: Optional[LLM] = None,
        language: str = "en",
    ):
        templates = PromptTemplateRegistry.get_templates("agent", language)

        goal_block = templates.get("goal_block", "").format(goal=goal) if goal else ""
        system_prompt = templates.get("system_prompt", "").format(
            role=role,
            background=background,
            goal_block=goal_block
        )
        description = templates.get("description", "").format(role=role)

        self._role = role
        self._background = background
        self._goal = goal

        super().__init__(
            name=name,
            tools=tools,
            tool_retriever=tool_retriever,
            can_handoff_to=can_handoff_to,
            llm=llm or Settings.llm,
            system_prompt=system_prompt,
            description=description,
        )
