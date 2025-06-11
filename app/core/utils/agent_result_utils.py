

from typing import List, Optional, Any
from llama_index.core.agent.workflow.workflow_events import (
    ToolCallResult,
    AgentOutput
)

from app.models.agent_models import SavantAgentOutput, SavantChatMessage, ToolResult, ToolResultOutput
from llama_index.core.workflow import Context


class AgentResultUtils:
    """Utility methods for handling ToolCallResult list."""

    @staticmethod
    async def parse_result_output(
        ctx:Context,
        agent_response:AgentOutput,
    ) -> Optional[SavantAgentOutput]:
        session_id = await ctx.get("session_id")
        conversation_id = await ctx.get("conversation_id")
        response = agent_response.response
        message = SavantChatMessage(content=response.content, role=response.role)
        tools_results: list[ToolResult] = []
        tool_call_results = agent_response.tool_calls or []
        for result in reversed(tool_call_results):
            tool_output = result.tool_output
            raw_output = tool_output.raw_output
            tool_name = tool_output.tool_name
            if raw_output :
                if isinstance(raw_output, ToolResultOutput):
                    tool_name = raw_output.tool_name
                else :    
                    raw_output = ToolResultOutput(content=raw_output, content_type="STRING", 
                                        conversation_id=conversation_id, session_id=session_id,  tool_name="DEFAULT")
                
            tools_result = ToolResult(tool_input=tool_output.raw_input , tool_name=tool_name, tool_output=raw_output, content=tool_output.content )
            tools_results.append(tools_result)
            
        
        return SavantAgentOutput(response=message, tools_results=tools_results)



    @staticmethod
    def get_raw_output(
        tool_call_results: List[ToolCallResult],
        tool_name: str,
    ) -> Optional[Any]:
        """
        Get the raw_output for the most recent (last) matching tool name.

        Args:
            tool_call_results (List[ToolCallResult]): List of ToolCallResult objects.
            tool_name (str): The name of the tool to look for.

        Returns:
            Optional[Any]: The raw_output value if found, else None.
        """
        for result in reversed(tool_call_results):
            if result.tool_name == tool_name:
                return result.tool_output.raw_output
        return None
