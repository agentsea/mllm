# MLLM

MultiModal Large Language Models

## Installation

```sh
pip install mllm
```

## Usage

Create an MLLM router from the API keys found in the current system env vars

```python
import os
from mllm import MLLMRouter

os.environ["OPENAI_API_KEY"] = "..."
os.environ["ANTHROPIC_API_KEY"] = "..."
os.environ["GEMINI_API_KEY"] = "..."

router = MLLMRouter.from_env()
```

Create a new role based chat thread

```python
from mllm import RoleThread

thread = RoleThread()
thread.post(role="user", msg="How are you?", images=["data:image/jpeg;base64,..."])
```

Chat with the MLLM, store the prompt data in the namespace `foo`

```python
response = router.chat(thread, namespace="foo")
thread.add_msg(response.msg)
```

Ask for a structured response

```python
from pydantic import BaseModel

class Foo(BaseModel):
    bar: str
    baz: int

thread.post(
    role="user",
    msg=f"What are bar and baz in this image? Please output as schema {Foo.model_json_schema()}"
    images=["data:image/jpeg;base64,..."]
)

response = router.chat(thread, namespace="foo", response_schema=Foo)
foo_parsed = response.parsed

assert type(foo_parsed) == Foo
```

Find a saved thread or a prompt

```python
RoleThread.find(id="123")
Prompt.find(id="456)
```

Just store prompts

```python
from mllm import Prompt, RoleThread

thread = RoleThread()

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
thread.add_msg(role_message)

response = call_openai(thread.to_openai())
response_msg = RoleMessage.from_openai(response["choices"][0]["message"])

saved_prompt = Prompt(thread, response_msg, namespace="foo")
```
