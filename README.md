# yapr

yapr (Yet Another Prompt)

## Installation

```sh
pip install yapr
```

## Usage

Create an LLM provider from the API keys found in the current system env vars

```python
from yapr import LLMProvider, RoleThread

llm_provider = LLMProvider.from_env()
```

Create a new role based chat thread

```python
thread = RoleThread()
thread.post(role="user", msg="How are you?")
```

Chat with the LLM, store the prompt data in the namespace "foo"

```python
response = llm_provider.chat(thread, namespace="foo")

thread.add_msg(response.msg)
```

Ask for a structured response

```python
from pydantic import BaseModel

class Foo(BaseModel):
    bar: str
    baz: int

thread.post(role="user", msg="Given the {...} can you return that in JSON?")

response = llm_provider.chat(thread, namespace="foo", response_schema=Foo)
foo_parsed = response.parsed

assert type(foo_parsed) == Foo
```

Multimodal

```python

```

Just store prompts

```python
from yapr import Prompt, RoleThread

msg = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Whats in this image?",
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,..."},
        }
    ]
}
role_message = RoleMessage.from_openai(msg)

thread = RoleThread()
thread.add_msg(role_message)

response = call_openai(thread.to_openai())

response_msg = RoleMessage.from_openai(response["choices"][0]["message"])

saved_prompt = Prompt(thread, response_msg, namespace="foo")
```
