# MLLM

MultiModal Large Language Models

## Installation

```sh
pip install mllm
```

## Usage

Create an MLLM router with a list of preferred models

```python
import os
from mllm import MLLMRouter

os.environ["OPENAI_API_KEY"] = "..."
os.environ["ANTHROPIC_API_KEY"] = "..."
os.environ["GEMINI_API_KEY"] = "..."

router = MLLMRouter(
    preference=["gpt-4-turbo", "anthropic/claude-3-opus-20240229", "gemini/gemini-pro-vision"]
)
```

Create a new role based chat thread

```python
from mllm import RoleThread

thread = RoleThread(owner_id="dolores@agentsea.ai")
thread.post(role="user", msg="Describe the image", images=["data:image/jpeg;base64,..."])
```

Chat with the MLLM, store the prompt data in the namespace `foo`

```python
response = router.chat(thread, namespace="foo")
thread.add_msg(response.msg)
```

Ask for a structured response

```python
from pydantic import BaseModel

class Animal(BaseModel):
    species: str
    color: str

thread.post(
    role="user",
    msg=f"What animal is in this image? Please output as schema {Animal.model_json_schema()}"
    images=["data:image/jpeg;base64,..."]
)

response = router.chat(thread, namespace="animal", response_schema=Animal)
animal_parsed = response.parsed

assert type(animal_parsed) == Animal
```

Find a saved thread or a prompt

```python
RoleThread.find(id="123")
Prompt.find(id="456)
```

If you just need to store prompts

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
