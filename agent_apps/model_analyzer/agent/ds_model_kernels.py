

import logging
from app.core.agent.kernel.savant_agent_kernel import SavantFunctionAgentKernel
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import (
    step
)
from llama_index.core.workflow.context import Context

from llama_index.core.agent.workflow.workflow_events import (
    AgentInput
)

from app.core.agent.kernel.workflow_agent.savant_agent_workflow_event import SaventAgentSetup


class ModelAnalyzerKernel(SavantFunctionAgentKernel):
    
    @step
    async def savant_init_agent(self, ctx: Context, ev: AgentInput) -> SaventAgentSetup:
        is_agent_init = await ctx.get("agent_init_flag", default=None)
        if not is_agent_init:
            logging.info("Initialized agent")
            await ctx.set("agent_init_flag", True)
            session_id = await ctx.get("session_id", default=None)
            if not session_id:
                error_msg = ChatMessage(
                    role="assistant",
                    content="Error : Model analyzer session is not initialized ",
                )
                ev.input.append(error_msg)
        return SaventAgentSetup(input=ev.input, current_agent_name=ev.current_agent_name)