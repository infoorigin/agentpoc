import difflib
import asyncio
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
    AsyncGenerator,
    Callable,
    Awaitable,
)

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponse,
    CompletionResponseGen,
    CompletionResponseAsyncGen,
    LLMMetadata,
    MessageRole
)
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback

from llama_index.core.tools.types import BaseTool
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.types import PydanticProgramMode
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    acompletion_to_chat_decorator,
    astream_chat_to_completion_decorator,
    astream_completion_to_chat_decorator,
    chat_to_completion_decorator,
    completion_to_chat_decorator,
    stream_chat_to_completion_decorator,
    stream_completion_to_chat_decorator,
)
from rapidfuzz import fuzz, process

class MockFunctionCallingLLM(FunctionCallingLLM, CustomLLM):
    """
    A production-grade mock LLM that simulates function-calling capabilities,
    tool invocation, and standard LLM endpoints for both chat and completion.
    Useful for testing agent pipelines and workflows.
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Any] = None,
        completion_to_prompt: Optional[Any] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
    ) -> None:
        super().__init__(
            max_tokens=max_tokens,
            callback_manager=callback_manager or CallbackManager([]),
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
        )

    @classmethod
    def class_name(cls) -> str:
        return "MockFunctionCallingLLM"

    @property
    def max_tokens(self) -> int:
        return 128000

    async def astream_chat_with_tools(
    self,
    tools: Sequence["BaseTool"],
    user_msg: Optional[Union[str, ChatMessage]] = None,
    chat_history: Optional[List[ChatMessage]] = None,
    verbose: bool = False,
    allow_parallel_tool_calls: bool = False,
    **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """
        Inline async stream chat with simulated token-wise output.
        """
        if isinstance(user_msg, str):
            resp_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        
        for chat_message in  chat_history:
            if chat_message.role == MessageRole.USER:
                resp_msg = chat_message
            if chat_message.role == MessageRole.TOOL:
                resp_msg = ChatMessage(role=MessageRole.ASSISTANT, content="Tool Executed Completed")


        chat_kwargs = self._prepare_chat_with_tools(
            tools,
            user_msg=user_msg,
            chat_history=chat_history,
            verbose=verbose,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )

        chat_response = ChatResponse(
            message=resp_msg,
            additional_kwargs=chat_kwargs
        )
        # Simulated response content
        response_text = self._generate_text(self.max_tokens)
        tokens = "m"

        async def gen() -> AsyncGenerator[ChatResponse, None]:
            accumulated = ""
            for token in tokens:
                accumulated += token + " "
                yield chat_response
                await asyncio.sleep(0)

        return gen()

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            is_function_calling_model=True,
            num_output=self.max_tokens or -1,
        )

    def _generate_text(self, length: int) -> str:
        return " ".join(["text"] * length)

    def _use_chat_completions(self, kwargs: Dict[str, Any]) -> bool:
        # Determine routing behavior. Customize as needed.
        return kwargs.get("mode", "chat") == "chat"

    def convert_user_msg(self, user_msg: Union[str, ChatMessage]) -> str:
        if isinstance(user_msg, ChatMessage) and user_msg.role == MessageRole.USER:
            return user_msg.content
        return ""

    def _prepare_chat_with_tools(
        self,
        tools: Sequence[BaseTool],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {
            "tools": tools,
            "user_msg": user_msg,
            "chat_history": chat_history or [],
            "verbose": verbose,
            "allow_parallel_tool_calls": allow_parallel_tool_calls,
        }

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolSelection]:
        response_kwargs = response.additional_kwargs
        tools: Sequence[BaseTool] = response_kwargs["tools"]
        user_msg: str = self.convert_user_msg(response.message)

        tool_descs = [tool.metadata.description for tool in tools]
        # best_match = difflib.get_close_matches(user_msg, tool_descs, n=1)

        best_match = process.extractOne(user_msg, tool_descs, scorer=fuzz.partial_ratio)

        if not user_msg or not best_match:
            if error_on_no_tool_call:
                raise ValueError("No suitable tool found for the given message.")
            return []

        matched_tool = next(tool for tool in tools if tool.metadata.description == best_match[0])
        return [ToolSelection(tool_id=matched_tool.metadata.name, tool_name=matched_tool.metadata.name, tool_kwargs={})]

    # === Chat Interfaces ===

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return ChatResponse(
            message=ChatMessage(role="assistant", content=self._generate_text(self.max_tokens or 5))
        )

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self.chat(messages, **kwargs)

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        for i in range(self.max_tokens or 5):
            yield ChatResponse(
                message=ChatMessage(role="assistant", content=self._generate_text(i + 1))
            )

    @llm_chat_callback()
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        astream_chat_fn: Callable[..., Awaitable[ChatResponseAsyncGen]]
        if self._use_chat_completions(kwargs):
            astream_chat_fn = self._astream_chat
        else:
            astream_chat_fn = astream_completion_to_chat_decorator(self._astream_complete)
        return await astream_chat_fn(messages, **kwargs)

    async def _astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        async def gen() -> AsyncGenerator[ChatResponse, None]:
            for i in range(self.max_tokens or 5):
                await asyncio.sleep(0)
                yield ChatResponse(
                    message=ChatMessage(role="assistant", content=self._generate_text(i + 1))
                )
        return gen()

    # === Completion Interfaces ===

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(
            text=self._generate_text(self.max_tokens or len(prompt.split()))
        )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        def gen() -> CompletionResponseGen:
            for i in range(self.max_tokens or 5):
                yield CompletionResponse(
                    text=self._generate_text(i + 1),
                    delta="text ",
                )
        return gen()

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return self.complete(prompt, formatted, **kwargs)

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        async def gen() -> AsyncGenerator[CompletionResponse, None]:
            for i in range(self.max_tokens or 5):
                await asyncio.sleep(0)
                yield CompletionResponse(
                    text=self._generate_text(i + 1),
                    delta="text ",
                )
        return gen()

    async def _astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        return await self.astream_complete(prompt, formatted, **kwargs)
