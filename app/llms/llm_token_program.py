from typing import Any, Dict, Optional, Tuple, Type, cast

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms.llm import LLM
from llama_index.core.output_parsers.pydantic import PydanticOutputParser
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.core.settings import Settings
from llama_index.core.types import BaseOutputParser, BasePydanticProgram

from app.llms.types import BasePydanticProgramWithTokenCount
from app.models.llm_models import LLMTokenCount
from app.utils.llm_counter_utils import get_token_count



class LLMTextCompletionProgramWithToken(BasePydanticProgramWithTokenCount[BaseModel]):
    """
    LLM Text Completion Program.

    Uses generic LLM text completion + an output parser to generate a structured output.

    """

    def __init__(
        self,
        output_parser: BaseOutputParser,
        output_cls: Type[BaseModel],
        prompt: BasePromptTemplate,
        llm: LLM,
        verbose: bool = False,
        generate_prompt = False
    ) -> None:
        self._output_parser = output_parser
        self._output_cls = output_cls
        self._llm = llm
        self._prompt = prompt
        self._verbose = verbose

        self._prompt.output_parser = output_parser
        self._generate_prompt = generate_prompt

    @classmethod
    def from_defaults(
        cls,
        output_parser: Optional[BaseOutputParser] = None,
        output_cls: Optional[Type[BaseModel]] = None,
        prompt_template_str: Optional[str] = None,
        prompt: Optional[BasePromptTemplate] = None,
        llm: Optional[LLM] = None,
        verbose: bool = False,
        generate_prompt = False,
        **kwargs: Any,
    ) -> "LLMTextCompletionProgramWithToken":
        llm = llm or Settings.llm
        if prompt is None and prompt_template_str is None:
            raise ValueError("Must provide either prompt or prompt_template_str.")
        if prompt is not None and prompt_template_str is not None:
            raise ValueError("Must provide either prompt or prompt_template_str.")
        if prompt_template_str is not None:
            prompt = PromptTemplate(prompt_template_str)

        # decide default output class if not set
        if output_cls is None:
            if not isinstance(output_parser, PydanticOutputParser):
                raise ValueError("Output parser must be PydanticOutputParser.")
            output_cls = output_parser.output_cls
        else:
            if output_parser is None:
                output_parser = PydanticOutputParser(output_cls=output_cls)

        return cls(
            output_parser,
            output_cls,
            prompt=cast(PromptTemplate, prompt),
            llm=llm,
            verbose=verbose,
            generate_prompt = generate_prompt
        )

    @property
    def output_cls(self) -> Type[BaseModel]:
        return self._output_cls

    @property
    def prompt(self) -> BasePromptTemplate:
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: BasePromptTemplate) -> None:
        self._prompt = prompt

    def __call__(
        self,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[BaseModel,LLMTokenCount]:
        llm_kwargs = llm_kwargs or {}
        if self._llm.metadata.is_chat_model:
            messages = self._prompt.format_messages(llm=self._llm, **kwargs)
            messages = self._llm._extend_messages(messages)
            message = messages[-1]
            formatted_prompt = message.content
            chat_response = self._llm.chat(messages, **llm_kwargs)

            raw_output = chat_response.message.content or ""
            additional_kwargs = chat_response.additional_kwargs
        else:
            formatted_prompt = self._prompt.format(llm=self._llm, **kwargs)

            response = self._llm.complete(formatted_prompt, **llm_kwargs)

            raw_output = response.text
            additional_kwargs = raw_output.additional_kwargs

        output = self._output_parser.parse(raw_output)
        token_count = get_token_count(**additional_kwargs)
        if not isinstance(output, self._output_cls):
            raise ValueError(
                f"Output parser returned {type(output)} but expected {self._output_cls}"
            )
        result = (output,token_count,formatted_prompt) if self._generate_prompt else (output,token_count)
        return result

    async def acall(
        self,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[BaseModel,LLMTokenCount]:
        llm_kwargs = llm_kwargs or {}
        if self._llm.metadata.is_chat_model:
            messages = self._prompt.format_messages(llm=self._llm, **kwargs)
            messages = self._llm._extend_messages(messages)
            message = messages[-1]
            formatted_prompt = message.content
            chat_response = await self._llm.achat(messages, **llm_kwargs)

            raw_output = chat_response.message.content or ""
            additional_kwargs = chat_response.additional_kwargs
        else:
            formatted_prompt = self._prompt.format(llm=self._llm, **kwargs)

            response = await self._llm.acomplete(formatted_prompt, **llm_kwargs)

            raw_output = response.text
            additional_kwargs = raw_output.additional_kwargs

        output = self._output_parser.parse(raw_output)
        token_count = get_token_count(**additional_kwargs)
        if not isinstance(output, self._output_cls):
            raise ValueError(
                f"Output parser returned {type(output)} but expected {self._output_cls}"
            )
        result = (output,token_count,formatted_prompt) if self._generate_prompt else (output,token_count)
        return result
