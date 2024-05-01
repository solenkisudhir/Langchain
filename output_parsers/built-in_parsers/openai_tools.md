# OpenAI Tools | 🦜️🔗 LangChain
These output parsers extract tool calls from OpenAI’s function calling API responses. This means they are only usable with models that support function calling, and specifically the latest `tools` and `tool_choice` parameters. We recommend familiarizing yourself with [function calling](https://python.langchain.com/docs/modules/model_io/chat/function_calling/) before reading this guide.

There are a few different variants of output parsers:

*   [JsonOutputToolsParser](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.openai_tools.JsonOutputToolsParser.html#langchain_core.output_parsers.openai_tools.JsonOutputToolsParser): Returns the arguments of the function call as JSON
*   [JsonOutputKeyToolsParser](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.openai_tools.JsonOutputKeyToolsParser.html#langchain_core.output_parsers.openai_tools.JsonOutputKeyToolsParser): Returns the value of specific key in the function call as JSON
*   [PydanticToolsParser](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.openai_tools.PydanticToolsParser.html#langchain_core.output_parsers.openai_tools.PydanticToolsParser): Returns the arguments of the function call as a Pydantic Model

```
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import ChatOpenAI

```


```
class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

```


```
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0).bind_tools([Joke])

```


```
[{'type': 'function',
  'function': {'name': 'Joke',
   'description': 'Joke to tell user.',
   'parameters': {'type': 'object',
    'properties': {'setup': {'description': 'question to set up a joke',
      'type': 'string'},
     'punchline': {'description': 'answer to resolve the joke',
      'type': 'string'}},
    'required': ['setup', 'punchline']}}}]

```


```
prompt = ChatPromptTemplate.from_messages(
    [("system", "You are helpful assistant"), ("user", "{input}")]
)

```


```
from langchain.output_parsers.openai_tools import JsonOutputToolsParser

```


```
parser = JsonOutputToolsParser()

```


```
chain = prompt | model | parser

```


```
chain.invoke({"input": "tell me a joke"})

```


```
[{'type': 'Joke',
  'args': {'setup': "Why don't scientists trust atoms?",
   'punchline': 'Because they make up everything!'}}]

```


To include the tool call id we can specify `return_id=True`:

```
parser = JsonOutputToolsParser(return_id=True)
chain = prompt | model | parser
chain.invoke({"input": "tell me a joke"})

```


```
[{'type': 'Joke',
  'args': {'setup': "Why don't scientists trust atoms?",
   'punchline': 'Because they make up everything!'},
  'id': 'call_Isuoh0RTeQzzOKGg5QlQ7UqI'}]

```


This merely extracts a single key from the returned response. This is useful for when you are passing in a single tool and just want it’s arguments.

```
from typing import List

from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser

```


```
parser = JsonOutputKeyToolsParser(key_name="Joke")

```


```
chain = prompt | model | parser

```


```
chain.invoke({"input": "tell me a joke"})

```


```
[{'setup': "Why don't scientists trust atoms?",
  'punchline': 'Because they make up everything!'}]

```


Certain models can return multiple tool invocations each call, so by default the output is a list. If we just want to return the first tool invocation, we can specify `first_tool_only=True`

```
parser = JsonOutputKeyToolsParser(key_name="Joke", first_tool_only=True)
chain = prompt | model | parser
chain.invoke({"input": "tell me a joke"})

```


```
{'setup': "Why don't scientists trust atoms?",
 'punchline': 'Because they make up everything!'}

```


This builds on top of `JsonOutputToolsParser` but passes the results to a Pydantic Model. This allows for further validation should you choose.

```
from langchain.output_parsers.openai_tools import PydanticToolsParser

```


```
class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    # You can add custom validation logic easily with Pydantic.
    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field


parser = PydanticToolsParser(tools=[Joke])

```


```
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0).bind_tools([Joke])
chain = prompt | model | parser

```


```
chain.invoke({"input": "tell me a joke"})

```


```
[Joke(setup="Why don't scientists trust atoms?", punchline='Because they make up everything!')]

```
