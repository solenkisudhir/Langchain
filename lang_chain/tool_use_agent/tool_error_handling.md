# Handling tool errors | 🦜️🔗 LangChain
Using a model to invoke a tool has some obvious potential failure modes. Firstly, the model needs to return a output that can be parsed at all. Secondly, the model needs to return tool arguments that are valid.

We can build error handling into our chains to mitigate these failure modes.

Setup[​](#setup "Direct link to Setup")
---------------------------------------

We’ll need to install the following packages:

```
%pip install --upgrade --quiet langchain-core langchain-openai

```


If you’d like to trace your runs in [LangSmith](https://python.langchain.com/docs/langsmith/) uncomment and set the following environment variables:

```
import getpass
import os

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

```


Chain[​](#chain "Direct link to Chain")
---------------------------------------

Suppose we have the following (dummy) tool and tool-calling chain. We’ll make our tool intentionally convoluted to try and trip up the model.

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
# Define tool
from langchain_core.tools import tool


@tool
def complex_tool(int_arg: int, float_arg: float, dict_arg: dict) -> int:
    """Do something complex with a complex tool."""
    return int_arg * float_arg

```


```
llm_with_tools = llm.bind_tools(
    [complex_tool],
)

```


```
# Define chain
chain = llm_with_tools | (lambda msg: msg.tool_calls[0]["args"]) | complex_tool

```


We can see that when we try to invoke this chain with even a fairly explicit input, the model fails to correctly call the tool (it forgets the `dict_arg` argument).

```
chain.invoke(
    "use complex tool. the args are 5, 2.1, empty dictionary. don't forget dict_arg"
)

```


```
ValidationError: 1 validation error for complex_toolSchema
dict_arg
  field required (type=value_error.missing)

```


The simplest way to more gracefully handle errors is to try/except the tool-calling step and return a helpful message on errors:

```
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig


def try_except_tool(tool_args: dict, config: RunnableConfig) -> Runnable:
    try:
        complex_tool.invoke(tool_args, config=config)
    except Exception as e:
        return f"Calling tool with arguments:\n\n{tool_args}\n\nraised the following error:\n\n{type(e)}: {e}"


chain = llm_with_tools | (lambda msg: msg.tool_calls[0]["args"]) | try_except_tool

```


```
print(
    chain.invoke(
        "use complex tool. the args are 5, 2.1, empty dictionary. don't forget dict_arg"
    )
)

```


```
Calling tool with arguments:

{'int_arg': 5, 'float_arg': 2.1}

raised the following error:

<class 'pydantic.v1.error_wrappers.ValidationError'>: 1 validation error for complex_toolSchema
dict_arg
  field required (type=value_error.missing)

```


Fallbacks[​](#fallbacks "Direct link to Fallbacks")
---------------------------------------------------

We can also try to fallback to a better model in the event of a tool invocation error. In this case we’ll fall back to an identical chain that uses `gpt-4-1106-preview` instead of `gpt-3.5-turbo`.

```
chain = llm_with_tools | (lambda msg: msg.tool_calls[0]["args"]) | complex_tool
better_model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0).bind_tools(
    [complex_tool], tool_choice="complex_tool"
)
better_chain = better_model | (lambda msg: msg.tool_calls[0]["args"]) | complex_tool

chain_with_fallback = chain.with_fallbacks([better_chain])
chain_with_fallback.invoke(
    "use complex tool. the args are 5, 2.1, empty dictionary. don't forget dict_arg"
)

```


Looking at the [Langsmith trace](https://smith.langchain.com/public/00e91fc2-e1a4-4b0f-a82e-e6b3119d196c/r) for this chain run, we can see that the first chain call fails as expected and it’s the fallback that succeeds.

Retry with exception[​](#retry-with-exception "Direct link to Retry with exception")
------------------------------------------------------------------------------------

To take things one step further, we can try to automatically re-run the chain with the exception passed in, so that the model may be able to correct its behavior:

```
import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough


class CustomToolException(Exception):
    """Custom LangChain tool exception."""

    def __init__(self, tool_call: ToolCall, exception: Exception) -> None:
        super().__init__()
        self.tool_call = tool_call
        self.exception = exception


def tool_custom_exception(msg: AIMessage, config: RunnableConfig) -> Runnable:
    try:
        return complex_tool.invoke(msg.tool_calls[0]["args"], config=config)
    except Exception as e:
        raise CustomToolException(msg.tool_calls[0], e)


def exception_to_messages(inputs: dict) -> dict:
    exception = inputs.pop("exception")

    # Add historical messages to the original input, so the model knows that it made a mistake with the last tool call.
    messages = [
        AIMessage(content="", tool_calls=[exception.tool_call]),
        ToolMessage(
            tool_call_id=exception.tool_call["id"], content=str(exception.exception)
        ),
        HumanMessage(
            content="The last tool call raised an exception. Try calling the tool again with corrected arguments. Do not repeat mistakes."
        ),
    ]
    inputs["last_output"] = messages
    return inputs


# We add a last_output MessagesPlaceholder to our prompt which if not passed in doesn't
# affect the prompt at all, but gives us the option to insert an arbitrary list of Messages
# into the prompt if needed. We'll use this on retries to insert the error message.
prompt = ChatPromptTemplate.from_messages(
    [("human", "{input}"), MessagesPlaceholder("last_output", optional=True)]
)
chain = prompt | llm_with_tools | tool_custom_exception

# If the initial chain call fails, we rerun it withe the exception passed in as a message.
self_correcting_chain = chain.with_fallbacks(
    [exception_to_messages | chain], exception_key="exception"
)

```


```
self_correcting_chain.invoke(
    {
        "input": "use complex tool. the args are 5, 2.1, empty dictionary. don't forget dict_arg"
    }
)

```


And our chain succeeds! Looking at the [LangSmith trace](https://smith.langchain.com/public/c11e804c-e14f-4059-bd09-64766f999c14/r), we can see that indeed our initial chain still fails, and it’s only on retrying that the chain succeeds.