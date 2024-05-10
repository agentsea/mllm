import os
import logging
from typing import Optional, List, Dict, TypeVar, Type, Generic
import time
import logging
from dataclasses import dataclass

from litellm import Router as LLMRouter, ModelResponse
from threadmem import RoleThread, RoleMessage
from litellm._logging import handler
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, before_sleep_log

from .models import V1EnvVarOpt, V1MLLMOption
from .util import extract_parse_json
from .prompt import Prompt

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


class Router:
    """
    A multimodal chat provider
    """

    provider_api_keys: Dict[str, str] = {
        "gpt-4-turbo": "OPENAI_API_KEY",
        "anthropic/claude-3-opus-20240229": "ANTHROPIC_API_KEY",
        "gemini/gemini-pro-vision": "GEMINI_API_KEY",
    }

    def __init__(
        self,
        preference: List[str] | str,
        timeout: int = 30,
        allow_fails: int = 1,
        num_retries: int = 3,
    ) -> None:
        self.model_list = []
        fallbacks = []

        if len(preference) == 0:
            raise Exception("No chat providers specified.")

        if isinstance(preference, str):
            preference = [preference]

        self.model = preference[0]

        # Construct the model list based on provided preferences and available API keys
        for provider in preference:
            api_key_env = self.provider_api_keys.get(provider)
            if api_key_env:
                api_key = os.getenv(api_key_env)
                if api_key:
                    self.model_list.append(
                        {
                            "model_name": provider,
                            "litellm_params": {
                                "model": provider,
                                "api_key": api_key,
                            },
                        }
                    )

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
        ) -> ChatResponse[T]:
            start = time.time()
            response = self.router.completion(model, thread.to_openai())

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

        return call_llm(thread, model, namespace, expect)

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
        available_providers = []

        preference_data = os.getenv("MODEL_PREFERENCE")
        preference = None
        if preference_data:
            preference = preference_data.split(",")
        if not preference:
            preference = cls.provider_api_keys.keys()

        logger.info(f"loading models with preference: {preference}")
        for provider in preference:
            env_var = cls.provider_api_keys.get(provider)
            if not env_var:
                raise ValueError(
                    f"Invalid provider '{provider}' specified in MODEL_PREFERENCE."
                )
            if os.getenv(env_var):
                logger.info(
                    f"Found LLM provider '{provider}' API key in environment variables."
                )
                available_providers.append(provider)

        if not available_providers:
            raise ValueError("No API keys found in environment variables.")

        return cls(available_providers)
