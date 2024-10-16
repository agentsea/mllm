import os
from typing import Generator, Union

import pytest
from pydantic import BaseModel
from threadmem import RoleThread

from mllm import ChatResponse, Prompt, Router, RouterConfig, StreamingResponseMessage


def test_router():
    router = Router.from_env()
    thread = RoleThread()

    class Animal(BaseModel):
        species: str
        color: str

    class Object(BaseModel):
        name: str
        color: str

    print("Schema: ", Animal.model_json_schema())
    thread.post(
        role="user",
        msg=(
            "can you describe whats in this image? "
            "Please output the response in raw JSON "
            f"in the following schema {Animal.model_json_schema()}"
        ),
        images=[
            "https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvZnJwYW50aGVyYV90aWdyaXNfYWx0YWljYV90aWdlcl8wLWltYWdlLWt6eGx2YzYyLmpwZw.jpg"
        ],
    )

    response = router.chat(thread, expect=Animal)

    assert type(response.parsed) == Animal
    print("\n\n!response: ", response)

    prompts = Prompt.find(id=response.prompt.id)
    assert len(prompts) == 1

    prompt = prompts[0]
    assert prompt.response.text == response.msg.text
    assert prompt.response_schema is not None
    assert prompt.response_schema == Animal.model_json_schema()

    thread = RoleThread()
    thread.post(
        role="user",
        msg=("can you describe whats in this image? "),
        images=[
            "https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvZnJwYW50aGVyYV90aWdyaXNfYWx0YWljYV90aWdlcl8wLWltYWdlLWt6eGx2YzYyLmpwZw.jpg"
        ],
    )

    response = router.chat(thread)

    print("\n\n!response: ", response)

    prompts = Prompt.find(id=response.prompt.id)
    assert len(prompts) == 1

    prompt = prompts[0]
    assert prompt.response.text == response.msg.text
    assert prompt.response_schema is None

    prompt.to_v1()

    thread.post(
        role="user",
        msg=(
            "can you describe whats in this image? "
            "Please output the response in raw JSON "
            f"in the following schema {Animal.model_json_schema()}"
        ),
        images=[
            "https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvZnJwYW50aGVyYV90aWdyaXNfYWx0YWljYV90aWdlcl8wLWltYWdlLWt6eGx2YzYyLmpwZw.jpg"
        ],
    )

    response = router.chat_multi(thread, expect_one=[Animal, Object])
    assert len(prompts) == 1
    assert type(response.parsed) == Animal


def test_router_stream_chat():
    router = Router.from_env()
    thread = RoleThread()

    class Animal(BaseModel):
        species: str
        color: str

    # Example of stream chat without any expected schema
    thread.post(
        role="user",
        msg="Can you describe the contents of this image?",
        images=[
            "https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvZnJwYW50aGVyYV90aWdyaXNfYWx0YWljYV90aWdlcl8wLWltYWdlLWt6eGx2YzYyLmpwZw.jpg"
        ],
    )

    stream_response: Generator[
        Union[StreamingResponseMessage, ChatResponse[BaseModel]], None, None
    ] = router.stream_chat(thread)

    parsed_response = None
    for chunk in stream_response:
        assert isinstance(chunk, StreamingResponseMessage) or isinstance(
            chunk, ChatResponse
        )
        if isinstance(chunk, ChatResponse):
            parsed_response = chunk

    assert parsed_response is not None
    assert isinstance(parsed_response, ChatResponse)
    assert parsed_response.msg.text is not None

    print("\nStreamed final response:", parsed_response.msg.text)

    prompts = Prompt.find(id=parsed_response.prompt.id)
    assert len(prompts) == 1

    prompt = prompts[0]
    assert prompt.response.text == parsed_response.msg.text
    assert prompt.response_schema is None  # In this case, no schema was provided

    # Example of stream chat with an expected schema
    thread.post(
        role="user",
        msg=(
            "Please describe the contents of the image in raw JSON format "
            f"according to this schema: {Animal.model_json_schema()}"
        ),
        images=[
            "https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvZnJwYW50aGVyYV90aWdyaXNfYWx0YWljYV90aWdlcl8wLWltYWdlLWt6eGx2YzYyLmpwZw.jpg"
        ],
    )

    stream_response = router.stream_chat(thread, expect=Animal)

    parsed_response = None
    for chunk in stream_response:
        assert isinstance(chunk, StreamingResponseMessage) or isinstance(
            chunk, ChatResponse
        )
        if isinstance(chunk, ChatResponse):
            parsed_response = chunk

    assert parsed_response is not None
    assert isinstance(parsed_response, ChatResponse)
    assert parsed_response.msg.text is not None
    assert isinstance(parsed_response.parsed, Animal)

    print("\nStreamed final response (with schema):", parsed_response.msg.text)

    prompts = Prompt.find(id=parsed_response.prompt.id)
    assert len(prompts) == 1

    prompt = prompts[0]
    assert prompt.response.text == parsed_response.msg.text
    assert prompt.response_schema == Animal.model_json_schema()


def test_router_chat_dynamic_model():
    router = Router("o1-mini")
    thread = RoleThread()

    thread.post(
        role="user",
        msg="Whats the capital of France?",
    )

    response = router.chat(thread)

    print("\nStreamed dynamic final response:", response.msg.text)


def test_router_custom_model():
    custom_model = RouterConfig(
        model="hosted_vllm/allenai/Molmo-7B-D-0924",
        api_base="https://models.agentlabs.xyz/v1",
        api_key_name="MOLMO_API_KEY",
    )
    router = Router(custom_model)
    thread = RoleThread()

    thread.post(
        role="user",
        msg="point to the statue",
        images=[
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        ],
    )

    response = router.chat(thread)
    print(f"\nCustom model response: {response.msg.text}")

    assert response.model == "hosted_vllm/allenai/Molmo-7B-D-0924"
    assert len(response.msg.text) > 0


def test_router_custom_model_missing_api_key():
    original_api_key = os.environ.get("MOLMO_API_KEY")

    if "MOLMO_API_KEY" in os.environ:
        del os.environ["MOLMO_API_KEY"]

    custom_model = RouterConfig(
        model="hosted_vllm/allenai/Molmo-7B-D-0924",
        api_base="https://models.agentlabs.xyz/v1",
        api_key_name="MOLMO_API_KEY",
    )

    with pytest.raises(Exception) as exc_info:
        Router(custom_model)

    assert "API key not found" in str(exc_info.value)

    if original_api_key is not None:
        os.environ["MOLMO_API_KEY"] = original_api_key


def test_router_mixed_models():
    custom_model = RouterConfig(
        model="hosted_vllm/allenai/Molmo-7B-D-0924",
        api_base="https://models.agentlabs.xyz/v1",
        api_key_name="MOLMO_API_KEY",
    )

    router = Router([custom_model, "gpt-4-turbo"])
    thread = RoleThread()

    # Test with custom model
    thread.post(
        role="user",
        msg="point to the statue",
        images=[
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        ],
    )

    response = router.chat(thread)
    print(f"\nMixed models - Custom model response: {response.msg.text}")

    assert response.model == "hosted_vllm/allenai/Molmo-7B-D-0924"
    assert len(response.msg.text) > 0

    # Test with standard model
    thread.post(role="user", msg="What's the capital of France?")

    response = router.chat(thread, model="gpt-4-turbo")
    print(f"\nMixed models - Standard model response: {response.msg.text}")

    assert response.model == "gpt-4-turbo-2024-04-09"
    assert "Paris" in response.msg.text


def test_router_unsupported_model():
    with pytest.raises(Exception):
        Router("unsupported_model")
