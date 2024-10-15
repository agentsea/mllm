from typing import Generator, Union

from threadmem import RoleThread
from pydantic import BaseModel

from mllm import Router, Prompt, ChatResponse, StreamingResponseMessage


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
