

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union
from app.core.agent.kernel.workflow_agent.multi_agent_workflow import AgentWorkflow, AgentWorkflowMeta
from llama_index.core.agent.workflow.base_agent import (
    BaseWorkflowAgent
)

from llama_index.core.workflow import (
    Context,
    step,
)

from llama_index.core.agent.workflow.workflow_events import (
    AgentInput,
    AgentSetup,
)

from app.core.agent.kernel.workflow_agent.savant_agent_workflow_event import SaventAgentSetup

FINAL_METHODS = {"setup_agent"}

class SavantAgentKernelMeta(AgentWorkflowMeta, ABCMeta):
    """Metaclass for AgentWorkflow that inherits from WorkflowMeta."""
    #TODO Use this step_methods = AgentWorkflow.get_steps_from_class(cls) to enforce that child class of SavantAgentWorkflow has
    # At least one @step method with input type ev: AgentInputand At least one @step method returning type AgentSetup
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace)

        # Skip check for the base class itself
        if namespace.get("__abstract__", False):
            return cls

        # Check for overridden final methods
        for method_name in FINAL_METHODS:
            if method_name in namespace:
                raise TypeError(
                    f"{name} is not allowed to override the final method `{method_name}` "
                    f"defined in SavantAgentWorkflow."
                )

        # (Optional) You can also use get_steps_from_class here to validate other constraints

        return cls

class SavantFunctionAgentKernel(AgentWorkflow, metaclass=SavantAgentKernelMeta):
    """A SavantAgentWorkflow for managing agent execution."""
    __abstract__ = True  # signal to metaclass not to validate this base

    def __init__(
        self,
        savant_agent: BaseWorkflowAgent,
        timeout: Optional[float] = None,
        **workflow_kwargs: Any,
    ):
        super().__init__(agents=[savant_agent], timeout=timeout, **workflow_kwargs)
        self.savant_agent = savant_agent

    @abstractmethod
    @step
    async def savant_init_agent(self, ctx: Context, ev: AgentInput) -> SaventAgentSetup:
        """This method must be implemented by subclasses."""
        pass

    @step
    async def setup_agent(self, ctx: Context, ev: SaventAgentSetup) -> AgentSetup:
        """Finalized step — cannot be overridden."""
        agent_setup: AgentSetup = await super().setup_agent(ctx, ev)
        return agent_setup

class DefaultSavantAgentWorkflow(SavantFunctionAgentKernel):
    
    @step
    async def savant_init_agent(self, ctx: Context, ev: AgentInput) -> SaventAgentSetup:
        await ctx.set("agent_init_flag", True)
        return SaventAgentSetup(input=ev.input, current_agent_name=ev.current_agent_name)    