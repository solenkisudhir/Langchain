# Parallel tool use | 🦜️🔗 LangChain
In the [Chains with multiple tools](https://python.langchain.com/docs/use_cases/tool_use/multiple_tools/) guide we saw how to build function-calling chains that select between multiple tools. Some models, like the OpenAI models released in Fall 2023, also support parallel function calling, which allows you to invoke multiple functions (or the same function multiple times) in a single model call. Our previous chain from the multiple tools guides actually already supports this.

If you’d like to trace your runs in [LangSmith](https://python.langchain.com/docs/langsmith/) uncomment and set the following environment variables:

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
[{'name': 'multiply',
  'args': {'first_int': 23, 'second_int': 7},
  'id': 'call_22tgOrsVLyLMsl2RLbUhtycw',
  'output': 161},
 {'name': 'multiply',
  'args': {'first_int': 5, 'second_int': 18},
  'id': 'call_EbKHEG3TjqBhEwb7aoxUtgzf',
  'output': 90},
 {'name': 'add',
  'args': {'first_int': 1000000, 'second_int': 1000000000},
  'id': 'call_LUhu2IT3vINxlTc5fCVY6Nhi',
  'output': 1001000000},
 {'name': 'exponentiate',
  'args': {'base': 37, 'exponent': 3},
  'id': 'call_bnCZIXelOKkmcyd4uGXId9Ct',
  'output': 50653}]

```
