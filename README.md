<!-- PROJECT LOGO -->
<br />
<p align="center">
  <!-- <a href="https://github.com/agentsea/skillpacks">
    <img src="https://project-logo.png" alt="Logo" width="80">
  </a> -->

  <h1 align="center">MLLM</h1>

  <p align="center">
    Multimodal Large Language Models
    <br />
    <a href="https://github.com/agentsea/mllm"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/agentsea/mllm">View Demo</a>
    ·
    <a href="https://github.com/agentsea/mllm/issues">Report Bug</a>
    ·
    <a href="https://github.com/agentsea/mllm/issues">Request Feature</a>
  </p>
  <br>
</p>

## Installation

```sh
pip install mllm
```

## Usage

Create an MLLM router with a list of preferred models

```python
import os
from mllm import Router

os.environ["OPENAI_API_KEY"] = "..."
os.environ["ANTHROPIC_API_KEY"] = "..."
os.environ["GEMINI_API_KEY"] = "..."

router = Router(
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

response = router.chat(thread, namespace="animal", expect=Animal)
animal_parsed = response.parsed

assert type(animal_parsed) == Animal
```

Find a saved thread or a prompt

```python
RoleThread.find(id="123")
Prompt.find(id="456)
```

To store a raw openai prompt

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

## Backends

Thread and prompt storage can be backed by:

- Sqlite
- Postgresql

Sqlite will be used by default. To use postgres simply configure the env vars:

```sh
DB_TYPE=postgres
DB_NAME=mllm
DB_HOST=localhost
DB_USER=postgres
DB_PASS=abc123
```
