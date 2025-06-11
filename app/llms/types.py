from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncGenerator,
    Generator,
    Generic,
    List,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from llama_index.core.bridge.pydantic import(
    BaseModel
)
from llama_index.core.instrumentation import DispatcherSpanMixin

from app.models.llm_models import LLMTokenCount

Model = TypeVar("Model", bound = BaseModel)

class BasePydanticProgramWithTokenCount(DispatcherSpanMixin, ABC, Generic[Model]):
    """A base class for llm-powered function that return a pydantic model.
    Note: this interface is not yet stable
    """

    @property
    @abstractmethod
    def output_cls(self) -> Type[Model]:
        pass

    @abstractmethod
    def __call__(self, *args: Any , **kwargs: Any) -> Tuple[Model,LLMTokenCount]:
        return self(*args, **kwargs)
    
    def stream_call(
            self, *args: Any, **kwargs: Any

    )-> Generator[Union[Model, List[Model]], None, None]:
        raise NotImplementedError("stream_call is not supported by default.")
    
    async def astream_call(
            self, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[Union[Model, List[Model]], None]:
        raise NotImplementedError("astream_call is not supported by default.")
    
    