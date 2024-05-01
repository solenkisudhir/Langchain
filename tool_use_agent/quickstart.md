# Quickstart | 🦜️🔗 LangChain
In this guide, we will go over the basic ways to create Chains and Agents that call Tools. Tools can be just about anything — APIs, functions, databases, etc. Tools allow us to extend the capabilities of a model beyond just outputting text/messages. The key to using models with tools is correctly prompting a model and parsing its response so that it chooses the right tools and provides the right inputs for them.

Setup[​](#setup "Direct link to Setup")
---------------------------------------

We’ll need to install the following packages for this guide:

```
%pip install --upgrade --quiet langchain

```


If you’d like to trace your runs in [LangSmith](https://python.langchain.com/docs/langsmith/) uncomment and set the following environment variables:

```
import getpass
import os

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

```


First, we need to create a tool to call. For this example, we will create a custom tool from a function. For more information on creating custom tools, please see [this guide](https://python.langchain.com/docs/modules/tools/).

```
from langchain_core.tools import tool


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

```


```
print(multiply.name)
print(multiply.description)
print(multiply.args)

```


```
multiply
multiply(first_int: int, second_int: int) -> int - Multiply two integers together.
{'first_int': {'title': 'First Int', 'type': 'integer'}, 'second_int': {'title': 'Second Int', 'type': 'integer'}}

```


```
multiply.invoke({"first_int": 4, "second_int": 5})

```


Chains[​](#chains "Direct link to Chains")
------------------------------------------

If we know that we only need to use a tool a fixed number of times, we can create a chain for doing so. Let’s create a simple chain that just multiplies user-specified numbers.

![chain](https://python.langchain.com/assets/images/tool_chain-3571e7fbc481d648aff93a2630f812ab.svg)

### Tool/function calling[​](#toolfunction-calling "Direct link to Tool/function calling")

One of the most reliable ways to use tools with LLMs is with tool calling APIs (also sometimes called function calling). This only works with models that explicitly support tool calling. You can see which models support tool calling [here](https://python.langchain.com/docs/integrations/chat/), and learn more about how to use tool calling in [this guide](https://python.langchain.com/docs/modules/model_io/chat/function_calling/).

First we’ll define our model and tools. We’ll start with just a single tool, `multiply`.

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


We’ll use `bind_tools` to pass the definition of our tool in as part of each call to the model, so that the model can invoke the tool when appropriate:

```
llm_with_tools = llm.bind_tools([multiply])

```


When the model invokes the tool, this will show up in the `AIMessage.tool_calls` attribute of the output:

```
msg = llm_with_tools.invoke("whats 5 times forty two")
msg.tool_calls

```


```
[{'name': 'multiply',
  'args': {'first_int': 5, 'second_int': 42},
  'id': 'call_cCP9oA3tRz7HDrjFn1FdmDaG'}]

```


Check out the [LangSmith trace here](https://smith.langchain.com/public/81ff0cbd-e05b-4720-bf61-2c9807edb708/r).

### Invoking the tool[​](#invoking-the-tool "Direct link to Invoking the tool")

Great! We’re able to generate tool invocations. But what if we want to actually call the tool? To do so we’ll need to pass the generated tool args to our tool. As a simple example we’ll just extract the arguments of the first tool\_call:

```
from operator import itemgetter

chain = llm_with_tools | (lambda x: x.tool_calls[0]["args"]) | multiply
chain.invoke("What's four times 23")

```


Check out the [LangSmith trace here](https://smith.langchain.com/public/16bbabb9-fc9b-41e5-a33d-487c42df4f85/r).

Agents[​](#agents "Direct link to Agents")
------------------------------------------

Chains are great when we know the specific sequence of tool usage needed for any user input. But for certain use cases, how many times we use tools depends on the input. In these cases, we want to let the model itself decide how many times to use tools and in what order. [Agents](https://python.langchain.com/docs/modules/agents/) let us do just this.

LangChain comes with a number of built-in agents that are optimized for different use cases. Read about all the [agent types here](https://python.langchain.com/docs/modules/agents/agent_types/).

We’ll use the [tool calling agent](https://python.langchain.com/docs/modules/agents/agent_types/tool_calling/), which is generally the most reliable kind and the recommended one for most use cases.

![agent](https://python.langchain.com/assets/images/tool_agent-d25fafc271da3ee950ac1fba59cdf490.svg)

```
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent

```


```
# Get the prompt to use - can be replaced with any prompt that includes variables "agent_scratchpad" and "input"!
prompt = hub.pull("hwchase17/openai-tools-agent")
prompt.pretty_print()

```


```
================================ System Message ================================

You are a helpful assistant

============================= Messages Placeholder =============================

{chat_history}

================================ Human Message =================================

{input}

============================= Messages Placeholder =============================

{agent_scratchpad}

```


Agents are also great because they make it easy to use multiple tools. To learn how to build Chains that use multiple tools, check out the [Chains with multiple tools](https://python.langchain.com/docs/use_cases/tool_use/multiple_tools/) page.

```
@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent


tools = [multiply, add, exponentiate]

```


```
# Construct the tool calling agent
agent = create_tool_calling_agent(llm, tools, prompt)

```


```
# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

```


With an agent, we can ask questions that require arbitrarily-many uses of our tools:

```
agent_executor.invoke(
    {
        "input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"
    }
)

```


```


> Entering new AgentExecutor chain...

Invoking: `exponentiate` with `{'base': 3, 'exponent': 5}`


243
Invoking: `add` with `{'first_int': 12, 'second_int': 3}`


15
Invoking: `multiply` with `{'first_int': 243, 'second_int': 15}`


3645
Invoking: `exponentiate` with `{'base': 405, 'exponent': 2}`


164025The result of taking 3 to the fifth power is 243. 

The sum of twelve and three is 15. 

Multiplying 243 by 15 gives 3645. 

Finally, squaring 3645 gives 164025.

> Finished chain.

```


```
{'input': 'Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result',
 'output': 'The result of taking 3 to the fifth power is 243. \n\nThe sum of twelve and three is 15. \n\nMultiplying 243 by 15 gives 3645. \n\nFinally, squaring 3645 gives 164025.'}

```


Check out the [LangSmith trace here](https://smith.langchain.com/public/eeeb27a4-a2f8-4f06-a3af-9c983f76146c/r).

Next steps[​](#next-steps "Direct link to Next steps")
------------------------------------------------------

Here we’ve gone over the basic ways to use Tools with Chains and Agents. We recommend the following sections to explore next:

*   [Agents](https://python.langchain.com/docs/modules/agents/): Everything related to Agents.
*   [Choosing between multiple tools](https://python.langchain.com/docs/use_cases/tool_use/multiple_tools/): How to make tool chains that select from multiple tools.
*   [Prompting for tool use](https://python.langchain.com/docs/use_cases/tool_use/prompting/): How to make tool chains that prompt models directly, without using function-calling APIs.
*   [Parallel tool use](https://python.langchain.com/docs/use_cases/tool_use/parallel/): How to make tool chains that invoke multiple tools at once.