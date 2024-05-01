# Using models that don't support tool calling | 🦜️🔗 LangChain
In this guide we’ll build a Chain that does not rely on any special model APIs (like tool calling, which we showed in the [Quickstart](https://python.langchain.com/docs/use_cases/tool_use/quickstart/)) and instead just prompts the model directly to invoke tools.

Setup[​](#setup "Direct link to Setup")
---------------------------------------

We’ll need to install the following packages:

```
%pip install --upgrade --quiet langchain langchain-openai

```


And set these environment variables:

```
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

# If you'd like to use LangSmith, uncomment the below:
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

```


First, we need to create a tool to call. For this example, we will create a custom tool from a function. For more information on all details related to creating custom tools, please see [this guide](https://python.langchain.com/docs/modules/tools/).

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


Creating our prompt[​](#creating-our-prompt "Direct link to Creating our prompt")
---------------------------------------------------------------------------------

We’ll want to write a prompt that specifies the tools the model has access to, the arguments to those tools, and the desired output format of the model. In this case we’ll instruct it to output a JSON blob of the form `{"name": "...", "arguments": {...}}`.

```
from langchain.tools.render import render_text_description

rendered_tools = render_text_description([multiply])
rendered_tools

```


```
'multiply: multiply(first_int: int, second_int: int) -> int - Multiply two integers together.'

```


```
from langchain_core.prompts import ChatPromptTemplate

system_prompt = f"""You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys."""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)

```


Adding an output parser[​](#adding-an-output-parser "Direct link to Adding an output parser")
---------------------------------------------------------------------------------------------

We’ll use the `JsonOutputParser` for parsing our models output to JSON.

```
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = prompt | model | JsonOutputParser()
chain.invoke({"input": "what's thirteen times 4"})

```


```
{'name': 'multiply', 'arguments': {'first_int': 13, 'second_int': 4}}

```


We can invoke the tool as part of the chain by passing along the model-generated “arguments” to it:

```
from operator import itemgetter

chain = prompt | model | JsonOutputParser() | itemgetter("arguments") | multiply
chain.invoke({"input": "what's thirteen times 4"})

```


Suppose we have multiple tools we want the chain to be able to choose from:

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


With function calling, we can do this like so:

If we want to run the model selected tool, we can do so using a function that returns the tool based on the model output. Specifically, our function will action return it’s own subchain that gets the “arguments” part of the model output and passes it to the chosen tool:

```
tools = [add, exponentiate, multiply]


def tool_chain(model_output):
    tool_map = {tool.name: tool for tool in tools}
    chosen_tool = tool_map[model_output["name"]]
    return itemgetter("arguments") | chosen_tool

```


```
rendered_tools = render_text_description(tools)
system_prompt = f"""You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys."""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)

chain = prompt | model | JsonOutputParser() | tool_chain
chain.invoke({"input": "what's 3 plus 1132"})

```


It can be helpful to return not only tool outputs but also tool inputs. We can easily do this with LCEL by `RunnablePassthrough.assign`\-ing the tool output. This will take whatever the input is to the RunnablePassrthrough components (assumed to be a dictionary) and add a key to it while still passing through everything that’s currently in the input:

```
from langchain_core.runnables import RunnablePassthrough

chain = (
    prompt | model | JsonOutputParser() | RunnablePassthrough.assign(output=tool_chain)
)
chain.invoke({"input": "what's 3 plus 1132"})

```


```
{'name': 'add',
 'arguments': {'first_int': 3, 'second_int': 1132},
 'output': 1135}

```
