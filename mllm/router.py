import logging
import os
import time
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from litellm import ModelResponse  # type: ignore
from litellm import Router as LLMRouter  # type: ignore
from litellm._logging import handler
from pydantic import BaseModel, Field
from tenacity import AsyncRetrying, before_sleep_log, retry, stop_after_attempt
from threadmem import RoleMessage, RoleThread

from .models import V1EnvVarOpt, V1LogitMetrics, V1MLLMOption
from .prompt import Prompt
from .util import extract_parse_json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

T = TypeVar("T", bound=BaseModel)


@dataclass
class ChatResponse(Generic[T]):
    model: str
    msg: RoleMessage
    time_elapsed: float
    tokens_request: int
    tokens_response: int
    prompt: Prompt
    parsed: Optional[T] = None
    logits: Optional[List[Dict[str, Any]]] = None
    logit_metrics: Optional[V1LogitMetrics] = None


@dataclass
class StreamingResponseMessage(Generic[T]):
    model: str
    msg: str
    time_elapsed: float
    tokens_response: int
    logits: Optional[List[Dict[str, Any]]] = None
    logit_metrics: Optional[V1LogitMetrics] = None


class RouterConfig(BaseModel):
    model: str
    api_base: Optional[str] = Field(default=None)
    api_key_name: Optional[str] = Field(default=None)


class Router:
    """
    A multimodal chat provider
    """

    provider_api_keys: Dict[str, str] = {
        "gpt-4o": "OPENAI_API_KEY",
        "gpt-4-turbo": "OPENAI_API_KEY",
        "gpt-4o-mini": "OPENAI_API_KEY",
        "anthropic/claude-3-opus-20240229": "ANTHROPIC_API_KEY",
        "anthropic/claude-3-5-sonnet-20240620": "ANTHROPIC_API_KEY",
        "gemini/gemini-1.5-pro-latest": "GEMINI_API_KEY",
        "gemini/gemini-pro": "GEMINI_API_KEY",
    }

    def __init__(
        self,
        preference: Union[
            List[str], str, List[RouterConfig], RouterConfig, List[str | RouterConfig]
        ],
        timeout: int = 30,
        allow_fails: int = 1,
        num_retries: int = 3,
    ) -> None:
        self.model_list = []

        if not preference:
            raise Exception("No chat providers specified.")

        if isinstance(preference, str) or isinstance(preference, RouterConfig):
            preference = [preference]  # type: ignore

        self.model = (
            preference[0] if isinstance(preference[0], str) else preference[0].model  # type: ignore
        )

        # Add models to model_list
        for item in preference:
            if isinstance(item, str):
                print(f"adding default model: {item}")
                self._add_default_model(item)
                print("added default model")
            elif isinstance(item, RouterConfig):
                self._add_custom_model(item)
            else:
                raise ValueError(f"Unsupported preference type: {type(item)}")

        if len(self.model_list) == 0:
            raise Exception("No valid API keys found for the specified providers.")

        # Create fallbacks list where each model falls back to the next model in the list
        fallbacks = []
        for i in range(len(self.model_list) - 1):
            fallbacks.append(
                {
                    self.model_list[i]["model_name"]: [
                        self.model_list[j]["model_name"]
                        for j in range(i + 1, len(self.model_list))
                    ]
                }
            )

        print(f"creating router with fallbacks: {fallbacks}")
        self.router = LLMRouter(
            model_list=self.model_list,
            timeout=timeout,
            allowed_fails=allow_fails,
            num_retries=num_retries,
            set_verbose=False,
            debug_level="INFO",
            fallbacks=fallbacks,
        )

        verbose_router_logger = logging.getLogger("LiteLLM Router")
        verbose_router_logger.setLevel(logging.ERROR)
        verbose_logger = logging.getLogger("LiteLLM")
        verbose_logger.setLevel(logging.ERROR)
        handler.setLevel(logging.ERROR)

    def _add_default_model(self, provider: str):
        api_key_env = self.provider_api_keys.get(provider)
        if api_key_env:
            provider_api_key = os.getenv(api_key_env)
            self.model_list.append(
                {
                    "model_name": provider,
                    "litellm_params": {
                        "model": provider,
                        "api_key": provider_api_key,
                    },
                }
            )
        else:
            self.model_list.append(
                {
                    "model_name": provider,
                    "litellm_params": {
                        "model": provider,
                    },
                }
            )

    def _add_custom_model(self, config: RouterConfig):
        if config.api_key_name:
            api_key = os.getenv(config.api_key_name)
            if not api_key:
                raise ValueError(
                    f"API key not found for environment variable: {config.api_key_name}"
                )
        else:
            api_key = None

        model_config = {
            "model_name": config.model,
            "litellm_params": {
                "model": config.model,
                "api_base": config.api_base,
            },
        }
        if api_key:
            model_config["litellm_params"]["api_key"] = api_key
        self.model_list.append(model_config)

    @classmethod
    def all_opts(cls) -> List[V1MLLMOption]:
        out = []
        for model, key in cls.provider_api_keys.items():
            out.append(
                V1MLLMOption(
                    model=model,
                    env_var=V1EnvVarOpt(
                        name=key,
                        description=f"{model} API key",
                        required=True,
                        secret=True,
                    ),
                )
            )
        return out

    def chat(
        self,
        thread: RoleThread,
        model: Optional[str] = None,
        namespace: str = "default",
        expect: Optional[Type[T]] = None,
        retries: int = 3,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_logprobs: int = 3,
        logprobs: bool = True,
        agent_id: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> ChatResponse[T]:
        """Chat with a language model

        Args:
            thread (RoleThread): A role thread
            model (Optional[str], optional): Model to use. Defaults to None.
            namespace (Optional[str], optional): Namespace to log into. Defaults to "default".
            expect (Optional[Type[T]], optional): Model type to expect response to conform to. Defaults to None.
            retries (int, optional): Number of retries if model fails. Defaults to 3.
            temperature (Optional[float], optional): Temperature for the model. Defaults to None.
            top_p (Optional[float], optional): Top P for the model. Defaults to None.
            top_logprobs (int, optional): Top logprobs for the model. Defaults to 3.
            logprobs (bool, optional): Whether to logprobs. Defaults to True.
            agent_id (Optional[str], optional): Agent ID for logging. Defaults to None.
            owner_id (Optional[str], optional): Owner ID for logging. Defaults to None.

        Returns:
            ChatResponse: A chat response
        """
        if not model:
            model = self.model  # type: ignore

        @retry(
            stop=stop_after_attempt(retries),
            before_sleep=before_sleep_log(logger, logging.ERROR),
        )
        def call_llm(
            thread: RoleThread,
            model: str,
            namespace: str = "default",
            expect: Optional[Type[T]] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            top_logprobs: int = 3,
            logprobs: bool = True,
        ) -> ChatResponse[T]:
            start = time.time()

            response = self.router.completion(
                model,
                thread.to_openai(),
                temperature=temperature,
                top_p=top_p,
                # top_logprobs=top_logprobs,
                # logprobs=logprobs,
                drop_params=True,
            )

            if not isinstance(response, ModelResponse):
                raise Exception(f"Unexpected response type: {type(response)}")

            end = time.time()
            elapsed = end - start

            logger.debug(f"llm response: {response.__dict__}")
            choices = response.choices[0]
            logits = None
            metrics = None
            if hasattr(choices, "logprobs"):
                resp_logprobs = choices.logprobs
                if resp_logprobs:
                    logits = resp_logprobs["content"]  # type: ignore
                    if logits:
                        metrics = self.calculate_logit_metrics(logits)

            response_obj = None
            msg = response["choices"][0]["message"].model_dump()
            content = msg["content"]
            if expect:
                try:
                    response_obj = expect.model_validate(extract_parse_json(content))
                except Exception as e:
                    logger.error(f"Validation error: {e} for '{content}")
                    raise

            resp_msg = RoleMessage(role=msg["role"], text=content)

            prompt = Prompt(
                thread=thread,
                response=resp_msg,
                response_schema=expect,  # type: ignore
                namespace=namespace,
                agent_id=agent_id,
                owner_id=owner_id,
                model=response.model or model,
                logits=logits,
                logit_metrics=metrics,
                temperature=temperature,
            )

            out = ChatResponse(
                model=response.model or model,
                msg=resp_msg,
                parsed=response_obj,
                time_elapsed=elapsed,
                tokens_request=response.usage.prompt_tokens,  # type: ignore
                tokens_response=response.usage.completion_tokens,  # type: ignore
                prompt=prompt,
                logits=logits,
                logit_metrics=metrics,
            )

            return out

        return call_llm(
            thread,
            model,  # type: ignore
            namespace,
            expect,
            temperature,
            top_p,
            top_logprobs,
            logprobs,
        )

    async def chat_async(
        self,
        thread: RoleThread,
        model: Optional[str] = None,
        namespace: str = "default",
        expect: Optional[Type[T]] = None,
        retries: int = 3,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_logprobs: int = 3,
        logprobs: bool = True,
        agent_id: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> ChatResponse[T]:
        """Chat asynchronously with a language model

        Args:
            thread (RoleThread): A role thread
            model (Optional[str], optional): Model to use. Defaults to None.
            namespace (Optional[str], optional): Namespace to log into. Defaults to "default".
            expect (Optional[Type[T]], optional): Model type to expect response to conform to. Defaults to None.
            retries (int, optional): Number of retries if model fails. Defaults to 3.
            temperature (Optional[float], optional): Temperature for the model. Defaults to None.
            top_p (Optional[float], optional): Top P for the model. Defaults to None.
            top_logprobs (int, optional): Top logprobs for the model. Defaults to 3.
            logprobs (bool, optional): Whether to logprobs. Defaults to True.
            agent_id (Optional[str], optional): Agent ID for logging. Defaults to None.
            owner_id (Optional[str], optional): Owner ID for logging. Defaults to None.

        Returns:
            ChatResponse: A chat response
        """
        if not model:
            model = self.model  # type: ignore

        async def call_llm(
            thread: RoleThread,
            model: str,
            namespace: str = "default",
            expect: Optional[Type[T]] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            top_logprobs: int = 3,
            logprobs: bool = True,
        ) -> ChatResponse[T]:
            start = time.time()

            response = await self.router.acompletion(
                model,
                thread.to_openai(),
                temperature=temperature,
                top_p=top_p,
                # top_logprobs=top_logprobs,
                # logprobs=logprobs,
                drop_params=True,
            )

            if not isinstance(response, ModelResponse):
                raise Exception(f"Unexpected response type: {type(response)}")

            end = time.time()
            elapsed = end - start

            logger.debug(f"llm response: {response.__dict__}")
            choices = response.choices[0]
            logits = None
            metrics = None
            if hasattr(choices, "logprobs"):
                resp_logprobs = choices.logprobs
                if resp_logprobs:
                    logits = resp_logprobs["content"]  # type: ignore
                    if logits:
                        metrics = self.calculate_logit_metrics(logits)

            response_obj = None
            msg = response["choices"][0]["message"].model_dump()
            content = msg["content"]
            if expect:
                try:
                    response_obj = expect.model_validate(extract_parse_json(content))
                except Exception as e:
                    logger.error(f"Validation error: {e} for '{content}")
                    raise

            resp_msg = RoleMessage(role=msg["role"], text=content)

            prompt = Prompt(
                thread=thread,
                response=resp_msg,
                response_schema=expect,  # type: ignore
                namespace=namespace,
                agent_id=agent_id,
                owner_id=owner_id,
                model=response.model or model,
                logits=logits,
                logit_metrics=metrics,
                temperature=temperature,
            )

            out = ChatResponse(
                model=response.model or model,
                msg=resp_msg,
                parsed=response_obj,
                time_elapsed=elapsed,
                tokens_request=response.usage.prompt_tokens,  # type: ignore
                tokens_response=response.usage.completion_tokens,  # type: ignore
                prompt=prompt,
                logits=logits,
                logit_metrics=metrics,
            )

            return out

        retrying = AsyncRetrying(
            stop=stop_after_attempt(retries),
            before_sleep=before_sleep_log(logger, logging.ERROR),
        )

        async for attempt in retrying:
            with attempt:
                return await call_llm(
                    thread,
                    model,  # type: ignore
                    namespace,
                    expect,
                    temperature,
                    top_p,
                    top_logprobs,
                    logprobs,
                )

        # Adding a fallback return here in case retries are exhausted
        raise Exception(
            "Retries exhausted: failed to get a valid response from the model."
        )

    def chat_multi(
        self,
        thread: RoleThread,
        model: Optional[str] = None,
        namespace: str = "default",
        expect_one: Optional[List[Type[BaseModel]]] = None,
        retries: int = 3,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_logprobs: int = 3,
        logprobs: bool = True,
        agent_id: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> ChatResponse[BaseModel]:
        """
        Chat with a language model expecting multiple possible types

        Args:
            thread (RoleThread): A role thread
            model (Optional[str], optional): Model to use. Defaults to None.
            namespace (str, optional): Namespace to log into. Defaults to "default".
            expect_one (List[Type[BaseModel]]): List of model types to expect, will return one of them.
            retries (int, optional): Number of retries if model fails. Defaults to 3.
            temperature (Optional[float], optional): Temperature for the model. Defaults to None.
            top_p (Optional[float], optional): Top P for the model. Defaults to None.
            top_logprobs (int, optional): Top logprobs for the model. Defaults to 3.
            logprobs (bool, optional): Logprobs for the model. Defaults to True.
            agent_id (Optional[str], optional): Agent ID for logging. Defaults to None.
            owner_id (Optional[str], optional): Owner ID for logging. Defaults to None.

        Returns:
            Tuple[ChatResponse[BaseModel], Type[BaseModel]]: A tuple containing the chat response and the type of the parsed object
        """
        if not model:
            model = self.model  # type: ignore

        if not expect_one:
            raise ValueError("At least one expected type must be provided")

        @retry(
            stop=stop_after_attempt(retries),
            before_sleep=before_sleep_log(logger, logging.ERROR),
        )
        def call_llm_multi(
            thread: RoleThread,
            model: str,
            namespace: str = "default",
            expect_one: Optional[List[Type[BaseModel]]] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            top_logprobs: int = 3,
            logprobs: bool = True,
        ) -> ChatResponse[BaseModel]:
            start = time.time()

            response = self.router.completion(
                model,
                thread.to_openai(),
                temperature=temperature,
                top_p=top_p,
                # top_logprobs=top_logprobs,
                # logprobs=logprobs,
                drop_params=True,
            )

            if not isinstance(response, ModelResponse):
                raise Exception(f"Unexpected response type: {type(response)}")

            end = time.time()

            elapsed = end - start

            logger.debug(f"llm response: {response.__dict__}")
            choices = response.choices[0]
            logits = None
            metrics = None
            if hasattr(choices, "logprobs"):
                resp_logprobs = choices.logprobs
                if resp_logprobs:
                    logits = resp_logprobs["content"]  # type: ignore
                    if logits:
                        metrics = self.calculate_logit_metrics(logits)

            msg = response["choices"][0]["message"].model_dump()
            content = msg["content"]

            response_obj = None
            matched_type = None
            if expect_one:
                for expected_type in expect_one:
                    try:
                        response_obj = expected_type.model_validate(
                            extract_parse_json(content)
                        )
                        matched_type = expected_type
                        break
                    except Exception:
                        continue

            if response_obj is None:
                logger.error(
                    f"Failed to validate against any of the expected types for '{content}'"
                )
                raise ValueError("Response did not match any of the expected types")

            resp_msg = RoleMessage(role=msg["role"], text=content)

            prompt = Prompt(
                thread=thread,
                response=resp_msg,
                response_schema=matched_type,  # type: ignore
                namespace=namespace,
                agent_id=agent_id,
                owner_id=owner_id,
                model=response.model or model,
                logits=logits,
                logit_metrics=metrics,
                temperature=temperature,
            )

            out = ChatResponse(
                model=response.model or model,
                msg=resp_msg,
                parsed=response_obj,
                time_elapsed=elapsed,
                tokens_request=response.usage.prompt_tokens,  # type: ignore
                tokens_response=response.usage.completion_tokens,  # type: ignore
                prompt=prompt,
                logits=logits,  # type: ignore
                logit_metrics=metrics,
            )

            return out

        return call_llm_multi(
            thread,
            model,  # type: ignore
            namespace,
            expect_one,
            temperature,
            top_p,
            top_logprobs,
            logprobs,
        )

    def check_model(self) -> None:
        """Check if the model is available"""

        thread = RoleThread()
        thread.post(
            "user", "Just checking if you are working... please return 'yes' if you are"
        )
        response = self.chat(thread)
        logger.debug(f"response from checking oai functionality: {response}")

    def options(self) -> List[V1MLLMOption]:
        """Dynamically generates options based on the configured providers."""
        options = []
        for model_info in self.model_list:
            model_name = model_info["model_name"]
            api_key_env = self.provider_api_keys.get(model_name)
            if api_key_env:
                option = V1MLLMOption(
                    model=model_name,
                    env_var=V1EnvVarOpt(
                        name=api_key_env,
                        description=f"{model_name} API key",
                        required=True,
                        secret=True,
                    ),
                )
                options.append(option)
        return options

    @classmethod
    def from_env(cls):
        """
        Class method to create an LLMProvider instance based on the API keys available in the environment variables.
        """
        preference_data = os.getenv("MODEL_PREFERENCE")
        if preference_data:
            preference = []
            for item in preference_data.split(","):
                parts = item.split("|")
                if len(parts) == 1:
                    preference.append(parts[0].strip())
                elif len(parts) == 3:
                    model, api_base, api_key_name = [p.strip() for p in parts]
                    preference.append(RouterConfig(model, api_base, api_key_name))  # type: ignore
                else:
                    raise ValueError(f"Invalid MODEL_PREFERENCE format: {item}")
        else:
            preference = list(cls.provider_api_keys.keys())

        return cls(preference)

    def stream_chat(
        self,
        thread: RoleThread,
        model: Optional[str] = None,
        namespace: str = "default",
        expect: Optional[Type[T]] = None,
        retries: int = 3,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_logprobs: int = 3,
        logprobs: bool = True,
        agent_id: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> Generator[Union[StreamingResponseMessage, ChatResponse[T]], None, None]:
        """
        Stream chat with a language model

        Args:
            thread (RoleThread): A role thread
            model (Optional[str], optional): Model to use. Defaults to None.
            namespace (str, optional): Namespace to log into. Defaults to "default".
            expect (Optional[Type[T]], optional): Model type to expect response to conform to. Defaults to None.
            retries (int, optional): Number of retries if model fails. Defaults to 3.
            temperature (Optional[float], optional): Temperature for the model. Defaults to None.
            top_p (Optional[float], optional): Top P for the model. Defaults to None.
            top_logprobs (int, optional): Top logprobs for the model. Defaults to 3.
            logprobs (bool, optional): Logprobs for the model. Defaults to True.
            agent_id (Optional[str], optional): Agent ID for logging. Defaults to None.
            owner_id (Optional[str], optional): Owner ID for logging. Defaults to None.

        Yields:
            Union[StreamingResponseMessage, ChatResponse[T]]: Streamed chat responses and final chat response
        """
        if not model:
            model = self.model  # type: ignore

        @retry(
            stop=stop_after_attempt(retries),
            before_sleep=before_sleep_log(logger, logging.ERROR),
        )
        def call_llm_stream(
            thread: RoleThread,
            model: str,
            namespace: str = "default",
            expect: Optional[Type[T]] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            top_logprobs: int = 3,
            logprobs: bool = True,
        ) -> Generator[Union[StreamingResponseMessage, ChatResponse[T]], None, None]:
            start = time.time()

            response = self.router.completion(
                model,
                thread.to_openai(),
                temperature=temperature,
                top_p=top_p,
                stream=True,
                # top_logprobs=top_logprobs,
                # logprobs=logprobs, # TODO: not supported by Gemini
                drop_params=True,
            )

            content = ""
            tokens_response = 0
            final_logits: List[Dict[str, Any]] = []

            for chunk in response:
                from litellm.types.utils import StreamingChoices

                choice: StreamingChoices = chunk.choices[0]  # type: ignore
                logits_dict = None
                metrics = None

                if hasattr(choice, "logprobs"):
                    resp_logprobs = choice.logprobs

                    if resp_logprobs:
                        logits = resp_logprobs.content  # type: ignore
                        if logits:
                            logits_dict = [logit.model_dump() for logit in logits]
                            metrics = self.calculate_logit_metrics(logits_dict)
                        final_logits.extend(logits_dict)  # type: ignore

                delta_content = choice.delta.content or ""  # type: ignore
                content += delta_content
                tokens_response += 1  # Approximate token count

                # Yield only the new chunk as a string
                streaming_response = StreamingResponseMessage(
                    model=model,
                    msg=delta_content,  # Changed to string
                    time_elapsed=time.time() - start,
                    tokens_response=tokens_response,
                    logits=logits_dict,
                    logit_metrics=metrics,
                )
                yield streaming_response

            # After streaming completes, attempt to parse the full content
            response_obj = None
            if expect:
                try:
                    response_obj = expect.model_validate(extract_parse_json(content))
                except Exception as e:
                    logger.error(f"Validation error: {e} for '{content}'")
                    raise

            final_metrics = self.calculate_logit_metrics(final_logits)

            # Create final ChatResponse with parsed content
            final_resp_msg = RoleMessage(role="assistant", text=content)
            prompt = Prompt(
                thread=thread,
                response=final_resp_msg,
                response_schema=expect,
                namespace=namespace,
                agent_id=agent_id,
                owner_id=owner_id,
                model=model,
                logits=final_logits,
                logit_metrics=final_metrics,
            )

            final_chat_response = ChatResponse(
                model=model,
                msg=final_resp_msg,
                parsed=response_obj if expect else None,
                time_elapsed=time.time() - start,
                tokens_request=0,  # Update if possible
                tokens_response=tokens_response,
                prompt=prompt,
                logits=final_logits,
                logit_metrics=final_metrics,
            )
            yield final_chat_response

        return call_llm_stream(
            thread,
            model,  # type: ignore
            namespace,
            expect,
            temperature,
            top_p,
            top_logprobs,
            logprobs,
        )

    async def stream_chat_async(
        self,
        thread: RoleThread,
        model: Optional[str] = None,
        namespace: str = "default",
        expect: Optional[Type[T]] = None,
        retries: int = 3,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_logprobs: int = 3,
        logprobs: bool = True,
        agent_id: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> AsyncGenerator[Union[StreamingResponseMessage, ChatResponse[T]], None]:
        """
        Stream chat with a language model

        Args:
            thread (RoleThread): A role thread
            model (Optional[str], optional): Model to use. Defaults to None.
            namespace (str, optional): Namespace to log into. Defaults to "default".
            expect (Optional[Type[T]], optional): Model type to expect response to conform to. Defaults to None.
            retries (int, optional): Number of retries if model fails. Defaults to 3.
            temperature (Optional[float], optional): Temperature for the model. Defaults to None.
            top_p (Optional[float], optional): Top P for the model. Defaults to None.
            top_logprobs (int, optional): Top logprobs for the model. Defaults to 3.
            logprobs (bool, optional): Logprobs for the model. Defaults to True.
            agent_id (Optional[str], optional): Agent ID for logging. Defaults to None.
            owner_id (Optional[str], optional): Owner ID for logging. Defaults to None.

        Yields:
            Union[StreamingResponseMessage, ChatResponse[T]]: Streamed chat responses and final chat response
        """
        if not model:
            model = self.model  # type: ignore

        @retry(
            stop=stop_after_attempt(retries),
            before_sleep=before_sleep_log(logger, logging.ERROR),
        )
        async def call_llm_stream(
            thread: RoleThread,
            model: str,
            namespace: str = "default",
            expect: Optional[Type[T]] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            top_logprobs: int = 3,
            logprobs: bool = True,
        ) -> AsyncGenerator[Union[StreamingResponseMessage, ChatResponse[T]], None]:
            start = time.time()

            # Assuming self.router.completion supports async operations
            response = await self.router.acompletion(
                model,
                thread.to_openai(),
                temperature=temperature,
                top_p=top_p,
                stream=True,  # Enable streaming
                # top_logprobs=top_logprobs,
                # logprobs=logprobs,
                drop_params=True,
            )

            content = ""
            tokens_response = 0
            final_logits: List[Dict[str, Any]] = []

            async for chunk in response:
                from litellm.types.utils import StreamingChoices

                choice: StreamingChoices = chunk.choices[0]  # type: ignore

                logits_dict = None
                metrics = None

                if hasattr(choice, "logprobs"):
                    resp_logprobs = choice.logprobs
                    if resp_logprobs:
                        logits = resp_logprobs.content  # type: ignore
                        if logits:
                            logits_dict = [logit.model_dump() for logit in logits]
                            metrics = self.calculate_logit_metrics(logits_dict)
                        final_logits.extend(logits_dict)  # type: ignore

                delta_content = choice.delta.content or ""  # type: ignore
                content += delta_content
                tokens_response += 1  # Approximate token count

                # Yield only the new chunk as a string
                streaming_response = StreamingResponseMessage(
                    model=model,
                    msg=delta_content,  # Changed to string
                    time_elapsed=time.time() - start,
                    tokens_response=tokens_response,
                    logits=logits_dict,
                    logit_metrics=metrics,
                )
                yield streaming_response

            # After streaming completes, attempt to parse the full content
            response_obj = None
            if expect:
                try:
                    response_obj = expect.model_validate(extract_parse_json(content))
                except Exception as e:
                    logger.error(f"Validation error: {e} for '{content}'")
                    raise

            # Create final ChatResponse with parsed content
            final_resp_msg = RoleMessage(role="assistant", text=content)
            prompt = Prompt(
                thread=thread,
                response=final_resp_msg,
                response_schema=expect,
                namespace=namespace,
                agent_id=agent_id,
                owner_id=owner_id,
                model=model,
            )

            final_chat_response = ChatResponse(
                model=model,
                msg=final_resp_msg,
                parsed=response_obj if expect else None,
                time_elapsed=time.time() - start,
                tokens_request=0,  # Update if possible
                tokens_response=tokens_response,
                prompt=prompt,
            )
            yield final_chat_response

        return call_llm_stream(
            thread,
            model,  # type: ignore
            namespace,
            expect,
            temperature,
            top_p,
            top_logprobs,
            logprobs,
        )

    def calculate_logit_metrics(self, logits: List[Dict[str, Any]]) -> V1LogitMetrics:
        entropies = []

        # Calculate entropy for each token
        for token_info in logits:
            # Calculate the entropy considering the top_logprobs
            token_entropies = []
            for top_prob_info in token_info["top_logprobs"]:
                logprob = top_prob_info["logprob"]
                # Since logprob is log(p), entropy is -p * log(p), where p = exp(logprob)
                p = np.exp(logprob)  # Convert logprob to probability
                entropy = -p * logprob
                token_entropies.append(entropy)

            # Aggregate the entropy for all top tokens
            total_entropy = np.sum(token_entropies)
            entropies.append(total_entropy)

        # Calculate average entropy
        average_entropy = float(np.mean(entropies))

        # Calculate variance of entropy (verentropy)
        varentropy = float(np.var(entropies))

        # Calculate derivative entropy
        derivative_entropies = np.diff(entropies).tolist()

        metrics = V1LogitMetrics(
            entropies=entropies,
            average_entropy=average_entropy,
            varentropy=varentropy,
            derivative_entropy=derivative_entropies,
        )

        return metrics
