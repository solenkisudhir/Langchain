# Human-in-the-loop | 🦜️🔗 LangChain
There are certain tools that we don’t trust a model to execute on its own. One thing we can do in such situations is require human approval before the tool is invoked.

Setup[​](#setup "Direct link to Setup")
---------------------------------------

We’ll need to install the following packages:

```
%pip install --upgrade --quiet langchain

```


And set these environment variables:

```
import getpass
import os

# If you'd like to use LangSmith, uncomment the below:
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

```


Chain[​](#chain "Direct link to Chain")
---------------------------------------

Suppose we have the following (dummy) tools and tool-calling chain:

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
from typing import Dict, List

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import tool


@tool
def count_emails(last_n_days: int) -> int:
    """Multiply two integers together."""
    return last_n_days * 2


@tool
def send_email(message: str, recipient: str) -> str:
    "Add two integers."
    return f"Successfully sent email to {recipient}."


tools = [count_emails, send_email]
llm_with_tools = llm.bind_tools(tools)


def call_tools(msg: AIMessage) -> List[Dict]:
    """Simple sequential tool calling helper."""
    tool_map = {tool.name: tool for tool in tools}
    tool_calls = msg.tool_calls.copy()
    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
    return tool_calls


chain = llm_with_tools | call_tools
chain.invoke("how many emails did i get in the last 5 days?")

```


```
[{'name': 'count_emails',
  'args': {'last_n_days': 5},
  'id': 'toolu_012VHuh7vk5dVNct5SgZj3gh',
  'output': 10}]

```


Adding human approval[​](#adding-human-approval "Direct link to Adding human approval")
---------------------------------------------------------------------------------------

We can add a simple human approval step to our tool\_chain function:

```
import json


def human_approval(msg: AIMessage) -> Runnable:
    tool_strs = "\n\n".join(
        json.dumps(tool_call, indent=2) for tool_call in msg.tool_calls
    )
    input_msg = (
        f"Do you approve of the following tool invocations\n\n{tool_strs}\n\n"
        "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no."
    )
    resp = input(input_msg)
    if resp.lower() not in ("yes", "y"):
        raise ValueError(f"Tool invocations not approved:\n\n{tool_strs}")
    return msg

```


```
chain = llm_with_tools | human_approval | call_tools
chain.invoke("how many emails did i get in the last 5 days?")

```


```
Do you approve of the following tool invocations

{
  "name": "count_emails",
  "args": {
    "last_n_days": 5
  },
  "id": "toolu_01LCpjpFxrRspygDscnHYyPm"
}

Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no. yes

```


```
[{'name': 'count_emails',
  'args': {'last_n_days': 5},
  'id': 'toolu_01LCpjpFxrRspygDscnHYyPm',
  'output': 10}]

```


```
chain.invoke("Send sally@gmail.com an email saying 'What's up homie'")

```


```
Do you approve of the following tool invocations

{
  "name": "send_email",
  "args": {
    "message": "What's up homie",
    "recipient": "sally@gmail.com"
  },
  "id": "toolu_0158qJVd1AL32Y1xxYUAtNEy"
}

Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no. no

```


```
ValueError: Tool invocations not approved:

{
  "name": "send_email",
  "args": {
    "message": "What's up homie",
    "recipient": "sally@gmail.com"
  },
  "id": "toolu_0158qJVd1AL32Y1xxYUAtNEy"
}

```
