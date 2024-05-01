# Choosing between multiple tools | 🦜️🔗 LangChain
In our [Quickstart](https://python.langchain.com/docs/use_cases/tool_use/quickstart/) we went over how to build a Chain that calls a single `multiply` tool. Now let’s take a look at how we might augment this chain so that it can pick from a number of tools to call. We’ll focus on Chains since [Agents](https://python.langchain.com/docs/use_cases/tool_use/agents/) can route between multiple tools by default.

Setup[​](#setup "Direct link to Setup")
---------------------------------------

We’ll need to install the following packages for this guide:

```
%pip install --upgrade --quiet langchain-core

```


If you’d like to trace your runs in [LangSmith](https://python.langchain.com/docs/langsmith/) uncomment and set the following environment variables:

```
import getpass
import os

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

```


Recall we already had a `multiply` tool:

```
from langchain_core.tools import tool


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

```


And now we can add to it an `exponentiate` and `add` tool:

```
@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent

```


The main difference between using one Tool and many is that we can’t be sure which Tool the model will invoke upfront, so we cannot hardcode, like we did in the [Quickstart](https://python.langchain.com/docs/use_cases/tool_use/quickstart/), a specific tool into our chain. Instead we’ll add `call_tools`, a `RunnableLambda` that takes the output AI message with tools calls and routes to the correct tools.

*   OpenAI
*   Anthropic
*   Google
*   Cohere
*   FireworksAI
*   MistralAI
*   TogetherAI

##### Install dependencies

```
pip install -qU langchain-openai

```


##### Set environment variables

```
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

```


```
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

```


```
from operator import itemgetter
from typing import Dict, List, Union

from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)

tools = [multiply, exponentiate, add]
llm_with_tools = llm.bind_tools(tools)
tool_map = {tool.name: tool for tool in tools}


def call_tools(msg: AIMessage) -> Runnable:
    """Simple sequential tool calling helper."""
    tool_map = {tool.name: tool for tool in tools}
    tool_calls = msg.tool_calls.copy()
    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
    return tool_calls


chain = llm_with_tools | call_tools

```


```
chain.invoke("What's 23 times 7")

```


```
[{'name': 'multiply',
  'args': {'first_int': 23, 'second_int': 7},
  'id': 'toolu_01Wf8kUs36kxRKLDL8vs7G8q',
  'output': 161}]

```


```
chain.invoke("add a million plus a billion")

```


```
[{'name': 'add',
  'args': {'first_int': 1000000, 'second_int': 1000000000},
  'id': 'toolu_012aK4xZBQg2sXARsFZnqxHh',
  'output': 1001000000}]

```


```
chain.invoke("cube thirty-seven")

```


```
[{'name': 'exponentiate',
  'args': {'base': 37, 'exponent': 3},
  'id': 'toolu_01VDU6X3ugDb9cpnnmCZFPbC',
  'output': 50653}]

```
