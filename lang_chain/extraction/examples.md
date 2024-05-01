# Use Reference Examples | 🦜️🔗 LangChain
[🦜️🔗](#)

*   [LangSmith](https://smith.langchain.com/)
*   [LangSmith Docs](https://docs.smith.langchain.com/)
*   [LangServe GitHub](https://github.com/langchain-ai/langserve)
*   [Templates GitHub](https://github.com/langchain-ai/langchain/tree/master/templates)
*   [Templates Hub](https://templates.langchain.com/)
*   [LangChain Hub](https://smith.langchain.com/hub)
*   [JS/TS Docs](https://js.langchain.com/)

[💬](https://chat.langchain.com/)[](https://github.com/langchain-ai/langchain)

*   [](https://python.langchain.com/)
*   [Use cases](https://python.langchain.com/docs/use_cases/)
*   [Extracting structured output](https://python.langchain.com/docs/use_cases/extraction/)
*   Use Reference Examples

Use Reference Examples
----------------------

The quality of extractions can often be improved by providing reference examples to the LLM.

While this tutorial focuses how to use examples with a tool calling model, this technique is generally applicable, and will work also with JSON more or prompt based techniques.

```
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked "
            "to extract, return null for the attribute's value.",
        ),
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        MessagesPlaceholder("examples"),  # <-- EXAMPLES!
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        ("human", "{text}"),
    ]
)

```


Test out the template:

```
from langchain_core.messages import (
    HumanMessage,
)

prompt.invoke(
    {"text": "this is some text", "examples": [HumanMessage(content="testing 1 2 3")]}
)

```


```
ChatPromptValue(messages=[SystemMessage(content="You are an expert extraction algorithm. Only extract relevant information from the text. If you do not know the value of an attribute asked to extract, return null for the attribute's value."), HumanMessage(content='testing 1 2 3'), HumanMessage(content='this is some text')])

```


Define the schema[​](#define-the-schema "Direct link to Define the schema")
---------------------------------------------------------------------------

Let’s re-use the person schema from the quickstart.

```
from typing import List, Optional

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(..., description="The name of the person")
    hair_color: Optional[str] = Field(
        ..., description="The color of the peron's eyes if known"
    )
    height_in_meters: Optional[str] = Field(..., description="Height in METERs")


class Data(BaseModel):
    """Extracted data about people."""

    # Creates a model so that we can extract multiple entities.
    people: List[Person]

```


Define reference examples[​](#define-reference-examples "Direct link to Define reference examples")
---------------------------------------------------------------------------------------------------

Examples can be defined as a list of input-output pairs.

Each example contains an example `input` text and an example `output` showing what should be extracted from the text.

This is a bit in the weeds, so feel free to ignore if you don’t get it!

The format of the example needs to match the API used (e.g., tool calling or JSON mode etc.).

Here, the formatted examples will match the format expected for the tool calling API since that’s what we’re using.

```
import uuid
from typing import Dict, List, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.pydantic_v1 import BaseModel, Field


class Example(TypedDict):
    """A representation of an example consisting of text input and expected tool calls.

    For extraction, the tool calls are represented as instances of pydantic model.
    """

    input: str  # This is the example text
    tool_calls: List[BaseModel]  # Instances of pydantic model that should be extracted


def tool_example_to_messages(example: Example) -> List[BaseMessage]:
    """Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts our example to a list of messages
    that can be fed into a chat model.

    The list of messages per example corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool correctly.

    The ToolMessage is required because some of the chat models are hyper-optimized for agents
    rather than for an extraction use case.
    """
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    openai_tool_calls = []
    for tool_call in example["tool_calls"]:
        openai_tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    # The name of the function right now corresponds
                    # to the name of the pydantic model
                    # This is implicit in the API right now,
                    # and will be improved over time.
                    "name": tool_call.__class__.__name__,
                    "arguments": tool_call.json(),
                },
            }
        )
    messages.append(
        AIMessage(content="", additional_kwargs={"tool_calls": openai_tool_calls})
    )
    tool_outputs = example.get("tool_outputs") or [
        "You have correctly called this tool."
    ] * len(openai_tool_calls)
    for output, tool_call in zip(tool_outputs, openai_tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return messages

```


Next let’s define our examples and then convert them into message format.

```
examples = [
    (
        "The ocean is vast and blue. It's more than 20,000 feet deep. There are many fish in it.",
        Person(name=None, height_in_meters=None, hair_color=None),
    ),
    (
        "Fiona traveled far from France to Spain.",
        Person(name="Fiona", height_in_meters=None, hair_color=None),
    ),
]


messages = []

for text, tool_call in examples:
    messages.extend(
        tool_example_to_messages({"input": text, "tool_calls": [tool_call]})
    )

```


Let’s test out the prompt

```
prompt.invoke({"text": "this is some text", "examples": messages})

```


```
ChatPromptValue(messages=[SystemMessage(content="You are an expert extraction algorithm. Only extract relevant information from the text. If you do not know the value of an attribute asked to extract, return null for the attribute's value."), HumanMessage(content="The ocean is vast and blue. It's more than 20,000 feet deep. There are many fish in it."), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'c75e57cc-8212-4959-81e9-9477b0b79126', 'type': 'function', 'function': {'name': 'Person', 'arguments': '{"name": null, "hair_color": null, "height_in_meters": null}'}}]}), ToolMessage(content='You have correctly called this tool.', tool_call_id='c75e57cc-8212-4959-81e9-9477b0b79126'), HumanMessage(content='Fiona traveled far from France to Spain.'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': '69da50b5-e427-44be-b396-1e56d821c6b0', 'type': 'function', 'function': {'name': 'Person', 'arguments': '{"name": "Fiona", "hair_color": null, "height_in_meters": null}'}}]}), ToolMessage(content='You have correctly called this tool.', tool_call_id='69da50b5-e427-44be-b396-1e56d821c6b0'), HumanMessage(content='this is some text')])

```


Create an extractor[​](#create-an-extractor "Direct link to Create an extractor")
---------------------------------------------------------------------------------

Here, we’ll create an extractor using **gpt-4**.

```
# We will be using tool calling mode, which
# requires a tool calling capable model.
llm = ChatOpenAI(
    # Consider benchmarking with a good model to get
    # a sense of the best possible quality.
    model="gpt-4-0125-preview",
    # Remember to set the temperature to 0 for extractions!
    temperature=0,
)


runnable = prompt | llm.with_structured_output(
    schema=Data,
    method="function_calling",
    include_raw=False,
)

```


```
/Users/harrisonchase/workplace/langchain/libs/core/langchain_core/_api/beta_decorator.py:86: LangChainBetaWarning: The function `with_structured_output` is in beta. It is actively being worked on, so the API may change.
  warn_beta(

```


Without examples 😿[​](#without-examples "Direct link to Without examples 😿")
------------------------------------------------------------------------------

Notice that even though we’re using gpt-4, it’s failing with a **very simple** test case!

```
for _ in range(5):
    text = "The solar system is large, but earth has only 1 moon."
    print(runnable.invoke({"text": text, "examples": []}))

```


```
people=[]
people=[Person(name='earth', hair_color=None, height_in_meters=None)]
people=[Person(name='earth', hair_color=None, height_in_meters=None)]
people=[]
people=[]

```


With examples 😻[​](#with-examples "Direct link to With examples 😻")
---------------------------------------------------------------------

Reference examples helps to fix the failure!

```
for _ in range(5):
    text = "The solar system is large, but earth has only 1 moon."
    print(runnable.invoke({"text": text, "examples": messages}))

```


```
people=[]
people=[]
people=[]
people=[]
people=[]

```


```
runnable.invoke(
    {
        "text": "My name is Harrison. My hair is black.",
        "examples": messages,
    }
)

```


```
Data(people=[Person(name='Harrison', hair_color='black', height_in_meters=None)])

```


* * *

#### Help us out by providing feedback on this documentation page:

[](https://python.langchain.com/docs/use_cases/extraction/guidelines/)[](https://python.langchain.com/docs/use_cases/extraction/how_to/handle_long_text/)

*   [Define the schema](#define-the-schema)
*   [Define reference examples](#define-reference-examples)
*   [Create an extractor](#create-an-extractor)
*   [Without examples 😿](#without-examples)
*   [With examples 😻](#with-examples)