# yapr

yapr (Yet Another Prompt)

## Installation

```sh
pip install yapr
```

## Usage

A complete chat example

```python
from yapr import LLMProvider, RoleThread

# Create an LLM provider from the API keys found in the current system env vars
llm_provider = LLMProvider.from_env()

# Create a new role based chat thread
thread = RoleThread()
thread.post(role="user", msg="How are you?")

# Chat with the LLM, store the prompt data in the namespace "foo"
response = llm_provider.chat(thread, namespace="foo")

# Add the response message to the thread
thread.add_msg(response.msg)

# Ask for a structured response
from pydantic import BaseModel

class Foo(BaseModel):
    bar: str
    baz: int

thread.post(role="user", msg="Given the {...} can you return that in JSON?")

# Chat with the LLM, requiring the output be parsable into the Foo object
response = llm_provider.chat(thread, namespace="foo", response_schema=Foo)

# Get the parsed response
foo_parsed = response.parsed
```
