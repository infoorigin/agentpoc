from llama_index.core.workflow import Event
from llama_index.core.llms import ChatMessage
from llama_index.core.agent.workflow.workflow_events import (
    AgentInput
)


class SaventAgentSetup(AgentInput):
    """Agent setup."""

   