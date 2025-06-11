from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Type, Union, TypeVar

from llama_index.core.llms import ChatMessage
from llama_index.core.memory import BaseMemory
from llama_index.core.workflow import (
    Context,
)
from llama_index.core.workflow.checkpointer import CheckpointCallback
from llama_index.core.workflow.handler import WorkflowHandler

from app.core.agent.kernel.savant_agent_kernel import DefaultSavantAgentWorkflow, SavantFunctionAgentKernel

T = TypeVar("T", bound="BaseWorkflowAgent")  # type: ignore[name-defined]


class SavantSingleAgentRunnerMixin(ABC):
    """
    Mixin class for executing a single agent within a workflow system.
    This class provides the necessary interface for running a single agent.
    """

    def _get_steps(self) -> Dict[str, Callable]:
        """Returns all the steps from the prebuilt workflow."""
        from app.core.agent.kernel.workflow_agent import AgentWorkflow

        instance = AgentWorkflow(agents=[self])  # type: ignore
        return instance._get_steps()

    def run(
        self,
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        ctx: Optional[Context] = None,
        stepwise: bool = False,
        checkpoint_callback: Optional[CheckpointCallback] = None,
        kernel_cls: Type[SavantFunctionAgentKernel] = DefaultSavantAgentWorkflow,
        **workflow_kwargs: Any,
    ) -> WorkflowHandler:
        """Run the agent."""
        from app.core.agent.kernel.workflow_agent import AgentWorkflow
        if not issubclass(kernel_cls, SavantFunctionAgentKernel):
            raise TypeError("workflow_cls must be a subclass of SavantAgentWorkflow")
        workflow = kernel_cls(savant_agent=self, **workflow_kwargs)  # type: ignore[list-item]
        agent_output =  workflow.run(
            user_msg=user_msg,
            chat_history=chat_history,
            memory=memory,
            ctx=ctx,
            stepwise=stepwise,
            checkpoint_callback=checkpoint_callback,
        )
        
        return agent_output
