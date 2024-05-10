from mllm import Router, Prompt
from threadmem import RoleThread
from pydantic import BaseModel


def test_router():
    router = Router.from_env()
    thread = RoleThread()

    class Animal(BaseModel):
        species: str
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

    prompts = Prompt.find(id=response.prompt_id)
    assert len(prompts) == 1

    prompt = prompts[0]
    assert prompt.response == response.msg
    assert prompt.response_schema is not None
    assert prompt.response_schema.model_json_schema() == Animal.model_json_schema()

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

    prompts = Prompt.find(id=response.prompt_id)
    assert len(prompts) == 1

    prompt = prompts[0]
    assert prompt.response == response.msg
    assert prompt.response_schema is None
