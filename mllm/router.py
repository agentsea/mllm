import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, Type, TypeVar, Generator, Union

from litellm import ModelResponse  # type: ignore
from litellm import Router as LLMRouter  # type: ignore
from litellm._logging import handler
from pydantic import BaseModel, Field
from tenacity import before_sleep_log, retry, stop_after_attempt
from threadmem import RoleMessage, RoleThread

from .models import V1EnvVarOpt, V1MLLMOption
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


@dataclass
class StreamingResponseMessage(Generic[T]):
    model: str
    msg: str
    time_elapsed: float
    tokens_response: int


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
    }
    def __init__(
        self,
        preference: Union[List[str], str, List[RouterConfig], RouterConfig],
        timeout: int = 30,
        allow_fails: int = 1,
        num_retries: int = 3,
    ) -> None:
        self.model_list = []
        fallbacks = []

        if not preference:
            raise Exception("No chat providers specified.")

        if isinstance(preference, str) or isinstance(preference, RouterConfig):
            preference = [preference]

        self.model = preference[0] if isinstance(preference[0], str) else preference[0].model

        for item in preference:
            if isinstance(item, str):
                self._add_default_model(item)
            elif isinstance(item, RouterConfig):
                self._add_custom_model(item)
            else:
                raise ValueError(f"Unsupported preference type: {type(item)}")

        if len(self.model_list) == 0:
            raise Exception("No valid API keys found for the specified providers.")

        # Calculate fallbacks dynamically
        for i, model in enumerate(self.model_list):
            fallback_models = self.model_list[i + 1 :]
            if fallback_models:
                fallbacks.append(
                    {
                        model["model_name"]: [
                            fallback["model_name"] for fallback in fallback_models
                        ]
                    }
                )

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
                raise ValueError(f"API key not found for environment variable: {config.api_key_name}")
        else:
            api_key = None

        model_config = {
            "model_name": config.model,
            "litellm_params": {
                "model": config.model,
                "api_base": config.api_base,
            }
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
            agent_id (Optional[str], optional): Agent ID for logging. Defaults to None.
            owner_id (Optional[str], optional): Owner ID for logging. Defaults to None.

        Returns:
            ChatResponse: A chat response
        """
        if not model:
            model = self.model

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
        ) -> ChatResponse[T]:
            start = time.time()

            response = self.router.completion(
                model,
                thread.to_openai(),
                temperature=temperature,
                top_p=top_p,
            )

            if not isinstance(response, ModelResponse):
                raise Exception(f"Unexpected response type: {type(response)}")

            end = time.time()

            elapsed = end - start

            logger.debug(f"llm response: {response.__dict__}")

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
            )

            out = ChatResponse(
                model=response.model or model,
                msg=resp_msg,
                parsed=response_obj,
                time_elapsed=elapsed,
                tokens_request=response.usage.prompt_tokens,  # type: ignore
                tokens_response=response.usage.completion_tokens,  # type: ignore
                prompt=prompt,
            )

            return out

        return call_llm(thread, model, namespace, expect, temperature, top_p)

    def chat_multi(
        self,
        thread: RoleThread,
        model: Optional[str] = None,
        namespace: str = "default",
        expect_one: Optional[List[Type[BaseModel]]] = None,
        retries: int = 3,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
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
            agent_id (Optional[str], optional): Agent ID for logging. Defaults to None.
            owner_id (Optional[str], optional): Owner ID for logging. Defaults to None.

        Returns:
            Tuple[ChatResponse[BaseModel], Type[BaseModel]]: A tuple containing the chat response and the type of the parsed object
        """
        if not model:
            model = self.model

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
        ) -> ChatResponse[BaseModel]:
            start = time.time()

            response = self.router.completion(
                model,
                thread.to_openai(),
                temperature=temperature,
                top_p=top_p,
            )

            if not isinstance(response, ModelResponse):
                raise Exception(f"Unexpected response type: {type(response)}")

            end = time.time()

            elapsed = end - start

            logger.debug(f"llm response: {response.__dict__}")

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
            )

            out = ChatResponse(
                model=response.model or model,
                msg=resp_msg,
                parsed=response_obj,
                time_elapsed=elapsed,
                tokens_request=response.usage.prompt_tokens,  # type: ignore
                tokens_response=response.usage.completion_tokens,  # type: ignore
                prompt=prompt,
            )

            return out

        return call_llm_multi(thread, model, namespace, expect_one, temperature, top_p)

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
                    preference.append(RouterConfig(model, api_base, api_key_name))
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
            agent_id (Optional[str], optional): Agent ID for logging. Defaults to None.
            owner_id (Optional[str], optional): Owner ID for logging. Defaults to None.

        Yields:
            Union[StreamingResponseMessage, ChatResponse[T]]: Streamed chat responses and final chat response
        """
        if not model:
            model = self.model

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
        ) -> Generator[Union[StreamingResponseMessage, ChatResponse[T]], None, None]:
            start = time.time()

            response = self.router.completion(
                model,
                thread.to_openai(),
                temperature=temperature,
                top_p=top_p,
                stream=True,  # Enable streaming
            )

            content = ""
            tokens_response = 0

            for chunk in response:
                delta_content = chunk.choices[0].delta.content or ""  # type: ignore
                content += delta_content
                tokens_response += 1  # Approximate token count

                # Yield only the new chunk as a string
                streaming_response = StreamingResponseMessage(
                    model=model,
                    msg=delta_content,  # Changed to string
                    time_elapsed=time.time() - start,
                    tokens_response=tokens_response,
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

        return call_llm_stream(thread, model, namespace, expect, temperature, top_p)

