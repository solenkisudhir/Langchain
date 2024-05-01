# Tool/function calling | 🦜️🔗 LangChain
info

We use the term tool calling interchangeably with function calling. Although function calling is sometimes meant to refer to invocations of a single function, we treat all models as though they can return multiple tool or function calls in each message.

Tool calling allows a model to respond to a given prompt by generating output that matches a user-defined schema. While the name implies that the model is performing some action, this is actually not the case! The model is coming up with the arguments to a tool, and actually running the tool (or not) is up to the user - for example, if you want to [extract output matching some schema](https://python.langchain.com/docs/use_cases/extraction/) from unstructured text, you could give the model an “extraction” tool that takes parameters matching the desired schema, then treat the generated output as your final result.

A tool call includes a name, arguments dict, and an optional identifier. The arguments dict is structured `{argument_name: argument_value}`.

Many LLM providers, including [Anthropic](https://www.anthropic.com/), [Cohere](https://cohere.com/), [Google](https://cloud.google.com/vertex-ai), [Mistral](https://mistral.ai/), [OpenAI](https://openai.com/), and others, support variants of a tool calling feature. These features typically allow requests to the LLM to include available tools and their schemas, and for responses to include calls to these tools. For instance, given a search engine tool, an LLM might handle a query by first issuing a call to the search engine. The system calling the LLM can receive the tool call, execute it, and return the output to the LLM to inform its response. LangChain includes a suite of [built-in tools](https://python.langchain.com/docs/integrations/tools/) and supports several methods for defining your own [custom tools](https://python.langchain.com/docs/modules/tools/custom_tools/). Tool-calling is extremely useful for building [tool-using chains and agents](https://python.langchain.com/docs/use_cases/tool_use/), and for getting structured outputs from models more generally.

Providers adopt different conventions for formatting tool schemas and tool calls. For instance, Anthropic returns tool calls as parsed structures within a larger content block:

```
[
  {
    "text": "<thinking>\nI should use a tool.\n</thinking>",
    "type": "text"
  },
  {
    "id": "id_value",
    "input": {"arg_name": "arg_value"},
    "name": "tool_name",
    "type": "tool_use"
  }
]

```


whereas OpenAI separates tool calls into a distinct parameter, with arguments as JSON strings:

```
{
  "tool_calls": [
    {
      "id": "id_value",
      "function": {
        "arguments": '{"arg_name": "arg_value"}',
        "name": "tool_name"
      },
      "type": "function"
    }
  ]
}

```


LangChain implements standard interfaces for defining tools, passing them to LLMs, and representing tool calls.

Chat models supporting tool calling features implement a `.bind_tools` method, which receives a list of LangChain [tool objects](https://api.python.langchain.com/en/latest/tools/langchain_core.tools.BaseTool.html#langchain_core.tools.BaseTool) and binds them to the chat model in its expected format. Subsequent invocations of the chat model will include tool schemas in its calls to the LLM.

For example, we can define the schema for custom tools using the `@tool` decorator on Python functions:

```
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]

```


Or below, we define the schema using Pydantic:

```
from langchain_core.pydantic_v1 import BaseModel, Field


# Note that the docstrings here are crucial, as they will be passed along
# to the model along with the class name.
class Add(BaseModel):
    """Add two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


class Multiply(BaseModel):
    """Multiply two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


tools = [Add, Multiply]

```


We can bind them to chat models as follows:

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


We can use the `bind_tools()` method to handle converting `Multiply` to a “tool” and binding it to the model (i.e., passing it in each time the model is invoked).

```
llm_with_tools = llm.bind_tools(tools)

```


If tool calls are included in a LLM response, they are attached to the corresponding [message](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.ai.AIMessage.html#langchain_core.messages.ai.AIMessage) or [message chunk](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.ai.AIMessageChunk.html#langchain_core.messages.ai.AIMessageChunk) as a list of [tool call](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.tool.ToolCall.html#langchain_core.messages.tool.ToolCall) objects in the `.tool_calls` attribute. A `ToolCall` is a typed dict that includes a tool name, dict of argument values, and (optionally) an identifier. Messages with no tool calls default to an empty list for this attribute.

Example:

```
query = "What is 3 * 12? Also, what is 11 + 49?"

llm_with_tools.invoke(query).tool_calls

```


```
[{'name': 'Multiply',
  'args': {'a': 3, 'b': 12},
  'id': 'call_1Tdp5wUXbYQzpkBoagGXqUTo'},
 {'name': 'Add',
  'args': {'a': 11, 'b': 49},
  'id': 'call_k9v09vYioS3X0Qg35zESuUKI'}]

```


The `.tool_calls` attribute should contain valid tool calls. Note that on occasion, model providers may output malformed tool calls (e.g., arguments that are not valid JSON). When parsing fails in these cases, instances of [InvalidToolCall](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.tool.InvalidToolCall.html#langchain_core.messages.tool.InvalidToolCall) are populated in the `.invalid_tool_calls` attribute. An `InvalidToolCall` can have a name, string arguments, identifier, and error message.

If desired, [output parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/) can further process the output. For example, we can convert back to the original Pydantic class:

```
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

chain = llm_with_tools | PydanticToolsParser(tools=[Multiply, Add])
chain.invoke(query)

```


```
[Multiply(a=3, b=12), Add(a=11, b=49)]

```


### Streaming[​](#streaming "Direct link to Streaming")

When tools are called in a streaming context, [message chunks](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.ai.AIMessageChunk.html#langchain_core.messages.ai.AIMessageChunk) will be populated with [tool call chunk](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.tool.ToolCallChunk.html#langchain_core.messages.tool.ToolCallChunk) objects in a list via the `.tool_call_chunks` attribute. A `ToolCallChunk` includes optional string fields for the tool `name`, `args`, and `id`, and includes an optional integer field `index` that can be used to join chunks together. Fields are optional because portions of a tool call may be streamed across different chunks (e.g., a chunk that includes a substring of the arguments may have null values for the tool name and id).

Because message chunks inherit from their parent message class, an [AIMessageChunk](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.ai.AIMessageChunk.html#langchain_core.messages.ai.AIMessageChunk) with tool call chunks will also include `.tool_calls` and `.invalid_tool_calls` fields. These fields are parsed best-effort from the message’s tool call chunks.

Note that not all providers currently support streaming for tool calls.

Example:

```
async for chunk in llm_with_tools.astream(query):
    print(chunk.tool_call_chunks)

```


```
[]
[{'name': 'Multiply', 'args': '', 'id': 'call_d39MsxKM5cmeGJOoYKdGBgzc', 'index': 0}]
[{'name': None, 'args': '{"a"', 'id': None, 'index': 0}]
[{'name': None, 'args': ': 3, ', 'id': None, 'index': 0}]
[{'name': None, 'args': '"b": 1', 'id': None, 'index': 0}]
[{'name': None, 'args': '2}', 'id': None, 'index': 0}]
[{'name': 'Add', 'args': '', 'id': 'call_QJpdxD9AehKbdXzMHxgDMMhs', 'index': 1}]
[{'name': None, 'args': '{"a"', 'id': None, 'index': 1}]
[{'name': None, 'args': ': 11,', 'id': None, 'index': 1}]
[{'name': None, 'args': ' "b": ', 'id': None, 'index': 1}]
[{'name': None, 'args': '49}', 'id': None, 'index': 1}]
[]

```


Note that adding message chunks will merge their corresponding tool call chunks. This is the principle by which LangChain’s various [tool output parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/types/openai_tools/) support streaming.

For example, below we accumulate tool call chunks:

```
first = True
async for chunk in llm_with_tools.astream(query):
    if first:
        gathered = chunk
        first = False
    else:
        gathered = gathered + chunk

    print(gathered.tool_call_chunks)

```


```
[]
[{'name': 'Multiply', 'args': '', 'id': 'call_erKtz8z3e681cmxYKbRof0NS', 'index': 0}]
[{'name': 'Multiply', 'args': '{"a"', 'id': 'call_erKtz8z3e681cmxYKbRof0NS', 'index': 0}]
[{'name': 'Multiply', 'args': '{"a": 3, ', 'id': 'call_erKtz8z3e681cmxYKbRof0NS', 'index': 0}]
[{'name': 'Multiply', 'args': '{"a": 3, "b": 1', 'id': 'call_erKtz8z3e681cmxYKbRof0NS', 'index': 0}]
[{'name': 'Multiply', 'args': '{"a": 3, "b": 12}', 'id': 'call_erKtz8z3e681cmxYKbRof0NS', 'index': 0}]
[{'name': 'Multiply', 'args': '{"a": 3, "b": 12}', 'id': 'call_erKtz8z3e681cmxYKbRof0NS', 'index': 0}, {'name': 'Add', 'args': '', 'id': 'call_tYHYdEV2YBvzDcSCiFCExNvw', 'index': 1}]
[{'name': 'Multiply', 'args': '{"a": 3, "b": 12}', 'id': 'call_erKtz8z3e681cmxYKbRof0NS', 'index': 0}, {'name': 'Add', 'args': '{"a"', 'id': 'call_tYHYdEV2YBvzDcSCiFCExNvw', 'index': 1}]
[{'name': 'Multiply', 'args': '{"a": 3, "b": 12}', 'id': 'call_erKtz8z3e681cmxYKbRof0NS', 'index': 0}, {'name': 'Add', 'args': '{"a": 11,', 'id': 'call_tYHYdEV2YBvzDcSCiFCExNvw', 'index': 1}]
[{'name': 'Multiply', 'args': '{"a": 3, "b": 12}', 'id': 'call_erKtz8z3e681cmxYKbRof0NS', 'index': 0}, {'name': 'Add', 'args': '{"a": 11, "b": ', 'id': 'call_tYHYdEV2YBvzDcSCiFCExNvw', 'index': 1}]
[{'name': 'Multiply', 'args': '{"a": 3, "b": 12}', 'id': 'call_erKtz8z3e681cmxYKbRof0NS', 'index': 0}, {'name': 'Add', 'args': '{"a": 11, "b": 49}', 'id': 'call_tYHYdEV2YBvzDcSCiFCExNvw', 'index': 1}]
[{'name': 'Multiply', 'args': '{"a": 3, "b": 12}', 'id': 'call_erKtz8z3e681cmxYKbRof0NS', 'index': 0}, {'name': 'Add', 'args': '{"a": 11, "b": 49}', 'id': 'call_tYHYdEV2YBvzDcSCiFCExNvw', 'index': 1}]

```


```
print(type(gathered.tool_call_chunks[0]["args"]))

```


And below we accumulate tool calls to demonstrate partial parsing:

```
first = True
async for chunk in llm_with_tools.astream(query):
    if first:
        gathered = chunk
        first = False
    else:
        gathered = gathered + chunk

    print(gathered.tool_calls)

```


```
[]
[]
[{'name': 'Multiply', 'args': {}, 'id': 'call_BXqUtt6jYCwR1DguqpS2ehP0'}]
[{'name': 'Multiply', 'args': {'a': 3}, 'id': 'call_BXqUtt6jYCwR1DguqpS2ehP0'}]
[{'name': 'Multiply', 'args': {'a': 3, 'b': 1}, 'id': 'call_BXqUtt6jYCwR1DguqpS2ehP0'}]
[{'name': 'Multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_BXqUtt6jYCwR1DguqpS2ehP0'}]
[{'name': 'Multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_BXqUtt6jYCwR1DguqpS2ehP0'}]
[{'name': 'Multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_BXqUtt6jYCwR1DguqpS2ehP0'}, {'name': 'Add', 'args': {}, 'id': 'call_UjSHJKROSAw2BDc8cp9cSv4i'}]
[{'name': 'Multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_BXqUtt6jYCwR1DguqpS2ehP0'}, {'name': 'Add', 'args': {'a': 11}, 'id': 'call_UjSHJKROSAw2BDc8cp9cSv4i'}]
[{'name': 'Multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_BXqUtt6jYCwR1DguqpS2ehP0'}, {'name': 'Add', 'args': {'a': 11}, 'id': 'call_UjSHJKROSAw2BDc8cp9cSv4i'}]
[{'name': 'Multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_BXqUtt6jYCwR1DguqpS2ehP0'}, {'name': 'Add', 'args': {'a': 11, 'b': 49}, 'id': 'call_UjSHJKROSAw2BDc8cp9cSv4i'}]
[{'name': 'Multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_BXqUtt6jYCwR1DguqpS2ehP0'}, {'name': 'Add', 'args': {'a': 11, 'b': 49}, 'id': 'call_UjSHJKROSAw2BDc8cp9cSv4i'}]

```


```
print(type(gathered.tool_calls[0]["args"]))

```


If we’re using the model-generated tool invocations to actually call tools and want to pass the tool results back to the model, we can do so using `ToolMessage`s.

```
from langchain_core.messages import HumanMessage, ToolMessage

messages = [HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)
for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_output = selected_tool.invoke(tool_call["args"])
    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
messages

```


```
[HumanMessage(content='What is 3 * 12? Also, what is 11 + 49?'),
 AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_K5DsWEmgt6D08EI9AFu9NaL1', 'function': {'arguments': '{"a": 3, "b": 12}', 'name': 'Multiply'}, 'type': 'function'}, {'id': 'call_qywVrsplg0ZMv7LHYYMjyG81', 'function': {'arguments': '{"a": 11, "b": 49}', 'name': 'Add'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 50, 'prompt_tokens': 105, 'total_tokens': 155}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_b28b39ffa8', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-1a0b8cdd-9221-4d94-b2ed-5701f67ce9fe-0', tool_calls=[{'name': 'Multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_K5DsWEmgt6D08EI9AFu9NaL1'}, {'name': 'Add', 'args': {'a': 11, 'b': 49}, 'id': 'call_qywVrsplg0ZMv7LHYYMjyG81'}]),
 ToolMessage(content='36', tool_call_id='call_K5DsWEmgt6D08EI9AFu9NaL1'),
 ToolMessage(content='60', tool_call_id='call_qywVrsplg0ZMv7LHYYMjyG81')]

```


```
llm_with_tools.invoke(messages)

```


```
AIMessage(content='3 * 12 is 36 and 11 + 49 is 60.', response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 171, 'total_tokens': 189}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_b28b39ffa8', 'finish_reason': 'stop', 'logprobs': None}, id='run-a6c8093c-b16a-4c92-8308-7c9ac998118c-0')

```


Few-shot prompting[​](#few-shot-prompting "Direct link to Few-shot prompting")
------------------------------------------------------------------------------

For more complex tool use it’s very useful to add few-shot examples to the prompt. We can do this by adding `AIMessage`s with `ToolCall`s and corresponding `ToolMessage`s to our prompt.

For example, even with some special instructions our model can get tripped up by order of operations:

```
llm_with_tools.invoke(
    "Whats 119 times 8 minus 20. Don't do any math yourself, only use tools for math. Respect order of operations"
).tool_calls

```


```
[{'name': 'Multiply',
  'args': {'a': 119, 'b': 8},
  'id': 'call_Dl3FXRVkQCFW4sUNYOe4rFr7'},
 {'name': 'Add',
  'args': {'a': 952, 'b': -20},
  'id': 'call_n03l4hmka7VZTCiP387Wud2C'}]

```


The model shouldn’t be trying to add anything yet, since it technically can’t know the results of 119 \* 8 yet.

By adding a prompt with some examples we can correct this behavior:

```
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

examples = [
    HumanMessage(
        "What's the product of 317253 and 128472 plus four", name="example_user"
    ),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {"name": "Multiply", "args": {"x": 317253, "y": 128472}, "id": "1"}
        ],
    ),
    ToolMessage("16505054784", tool_call_id="1"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[{"name": "Add", "args": {"x": 16505054784, "y": 4}, "id": "2"}],
    ),
    ToolMessage("16505054788", tool_call_id="2"),
    AIMessage(
        "The product of 317253 and 128472 plus four is 16505054788",
        name="example_assistant",
    ),
]

system = """You are bad at math but are an expert at using a calculator. 

Use past tool usage as an example of how to correctly use the tools."""
few_shot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        *examples,
        ("human", "{query}"),
    ]
)

chain = {"query": RunnablePassthrough()} | few_shot_prompt | llm_with_tools
chain.invoke("Whats 119 times 8 minus 20").tool_calls

```


```
[{'name': 'Multiply',
  'args': {'a': 119, 'b': 8},
  'id': 'call_MoSgwzIhPxhclfygkYaKIsGZ'}]

```


Seems like we get the correct output this time.

Here’s what the [LangSmith trace](https://smith.langchain.com/public/f70550a1-585f-4c9d-a643-13148ab1616f/r) looks like.

Next steps[​](#next-steps "Direct link to Next steps")
------------------------------------------------------

*   **Output parsing**: See [OpenAI Tools output parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/types/openai_tools/) and [OpenAI Functions output parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/types/openai_functions/) to learn about extracting the function calling API responses into various formats.
*   **Structured output chains**: [Some models have constructors](https://python.langchain.com/docs/modules/model_io/chat/structured_output/) that handle creating a structured output chain for you.
*   **Tool use**: See how to construct chains and agents that call the invoked tools in [these guides](https://python.langchain.com/docs/use_cases/tool_use/).