import os
from typing import AsyncGenerator, Union

import pytest
from pydantic import BaseModel
from threadmem import RoleThread

from mllm import (
    ChatResponse,
    Prompt,
    Router,
    RouterConfig,
    StreamingResponseMessage,
)


@pytest.mark.asyncio
async def test_router_chat_async():
    router = Router("gpt-4-turbo")
    thread = RoleThread()

    class Animal(BaseModel):
        species: str
        color: str

    print("Schema: ", Animal.model_json_schema())
    thread.post(
        role="user",
        msg=(
            "Can you describe the contents of this image? "
            "Please output a raw JSON object "
            f"following the schema: {Animal.model_json_schema()}"
            "For instance, if the image is a dog, the output should be: "
            "{'species': 'dog', 'color': 'brown'}"
        ),
        images=[
            "https://images.unsplash.com/photo-1551969014-7d2c4cddf0b6?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxleHBsb3JlLWZlZWR8MTB8fHxlbnwwfHx8fHw%3D"
        ],
    )

    response = await router.chat_async(thread, expect=Animal, retries=0)

    assert isinstance(response.parsed, Animal)
    print("\nAsync response:", response)

    prompts = Prompt.find(id=response.prompt.id)
    assert len(prompts) == 1

    prompt = prompts[0]
    assert prompt.response.text == response.msg.text
    assert prompt.response_schema == Animal.model_json_schema()


@pytest.mark.asyncio
async def test_router_chat_async_no_schema():
    router = Router("gpt-4-turbo")
    thread = RoleThread()

    thread.post(
        role="user",
        msg="What's the capital of France?",
    )

    response = await router.chat_async(thread)

    assert response.msg.text is not None
    print("\nAsync response without schema:", response.msg.text)

    prompts = Prompt.find(id=response.prompt.id)
    assert len(prompts) == 1

    prompt = prompts[0]
    assert prompt.response.text == response.msg.text
    assert prompt.response_schema is None


@pytest.mark.asyncio
async def test_router_stream_chat_async():
    router = Router.from_env()
    thread = RoleThread()

    class Animal(BaseModel):
        species: str
        color: str

    thread.post(
        role="user",
        msg=(
            "Please describe the contents of the image in raw JSON format "
            f"according to this schema: {Animal.model_json_schema()}"
        ),
        images=[
            "https://images.unsplash.com/photo-1551969014-7d2c4cddf0b6?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxleHBsb3JlLWZlZWR8MTB8fHxlbnwwfHx8fHw%3D"
        ],
    )

    stream_response = await router.stream_chat_async(
        thread, expect=Animal
    )  # Await the coroutine

    parsed_response = None
    async for chunk in stream_response:
        assert isinstance(chunk, (StreamingResponseMessage, ChatResponse))
        if isinstance(chunk, ChatResponse):
            parsed_response = chunk

    assert parsed_response is not None
    assert isinstance(parsed_response.parsed, Animal)

    print("\nStreamed async final response (with schema):", parsed_response.msg.text)

    prompts = Prompt.find(id=parsed_response.prompt.id)
    assert len(prompts) == 1

    prompt = prompts[0]
    assert prompt.response.text == parsed_response.msg.text
    assert prompt.response_schema == Animal.model_json_schema()


@pytest.mark.asyncio
async def test_router_stream_chat_async_no_schema():
    router = Router.from_env()
    thread = RoleThread()

    thread.post(
        role="user",
        msg="Can you tell me a joke?",
    )

    stream_response = await router.stream_chat_async(thread)

    parsed_response = None
    async for chunk in stream_response:
        assert isinstance(chunk, (StreamingResponseMessage, ChatResponse))
        if isinstance(chunk, ChatResponse):
            parsed_response = chunk

    assert parsed_response is not None
    assert parsed_response.msg.text is not None

    print("\nStreamed async final response without schema:", parsed_response.msg.text)

    prompts = Prompt.find(id=parsed_response.prompt.id)
    assert len(prompts) == 1

    prompt = prompts[0]
    assert prompt.response.text == parsed_response.msg.text
    assert prompt.response_schema is None


# @pytest.mark.asyncio
# async def test_router_chat_async_custom_model():
#     custom_model = RouterConfig(
#         model="hosted_vllm/allenai/Molmo-7B-D-0924",
#         api_base="https://models.agentlabs.xyz/v1",
#         api_key_name="MOLMO_API_KEY",
#     )
#     router = Router(custom_model)
#     thread = RoleThread()

#     thread.post(
#         role="user",
#         msg="Point to the statue",
#         images=[
#             "https://cdn.britannica.com/61/93061-050-99147DCE/"
#             "Statue-of-Liberty-Island-New-York-Bay.jpg"
#         ],
#     )

#     response = await router.chat_async(thread)
#     print(f"\nCustom model async response: {response.msg.text}")

#     assert response.model == "hosted_vllm/allenai/Molmo-7B-D-0924"
#     assert len(response.msg.text) > 0


# @pytest.mark.asyncio
# async def test_router_stream_chat_async_custom_model():
#     custom_model = RouterConfig(
#         model="hosted_vllm/allenai/Molmo-7B-D-0924",
#         api_base="https://models.agentlabs.xyz/v1",
#         api_key_name="MOLMO_API_KEY",
#     )
#     router = Router(custom_model)
#     thread = RoleThread()

#     thread.post(
#         role="user",
#         msg=(
#             "Please describe the contents of the image in raw JSON format "
#             "with keys 'description' and 'objects'."
#         ),
#         images=[
#             "https://cdn.britannica.com/61/93061-050-99147DCE/"
#             "Statue-of-Liberty-Island-New-York-Bay.jpg"
#         ],
#     )

#     stream_response = await router.stream_chat_async(thread)

#     parsed_response = None
#     async for chunk in stream_response:
#         assert isinstance(chunk, (StreamingResponseMessage, ChatResponse))
#         if isinstance(chunk, ChatResponse):
#             parsed_response = chunk

#     assert parsed_response is not None
#     assert len(parsed_response.msg.text) > 0
#     print("\nStreamed async custom model final response:", parsed_response.msg.text)

#     assert parsed_response.model == "hosted_vllm/allenai/Molmo-7B-D-0924"
