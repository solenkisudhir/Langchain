# Lambda: Run custom functions | 🦜️🔗 LangChain
You can use arbitrary functions in the pipeline.

Note that all inputs to these functions need to be a SINGLE argument. If you have a function that accepts multiple arguments, you should write a wrapper that accepts a single input and unpacks it into multiple argument.

%pip install –upgrade –quiet langchain langchain-openai

```
from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI


def length_function(text):
    return len(text)


def _multiple_length_function(text1, text2):
    return len(text1) * len(text2)


def multiple_length_function(_dict):
    return _multiple_length_function(_dict["text1"], _dict["text2"])


prompt = ChatPromptTemplate.from_template("what is {a} + {b}")
model = ChatOpenAI()

chain1 = prompt | model

chain = (
    {
        "a": itemgetter("foo") | RunnableLambda(length_function),
        "b": {"text1": itemgetter("foo"), "text2": itemgetter("bar")}
        | RunnableLambda(multiple_length_function),
    }
    | prompt
    | model
)

```


```
chain.invoke({"foo": "bar", "bar": "gah"})

```


```
AIMessage(content='3 + 9 = 12', response_metadata={'token_usage': {'completion_tokens': 7, 'prompt_tokens': 14, 'total_tokens': 21}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_b28b39ffa8', 'finish_reason': 'stop', 'logprobs': None}, id='run-bd204541-81fd-429a-ad92-dd1913af9b1c-0')

```


Accepting a Runnable Config[​](#accepting-a-runnable-config "Direct link to Accepting a Runnable Config")
---------------------------------------------------------------------------------------------------------

Runnable lambdas can optionally accept a [RunnableConfig](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.config.RunnableConfig.html#langchain_core.runnables.config.RunnableConfig), which they can use to pass callbacks, tags, and other configuration information to nested runs.

```
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

```


```
import json


def parse_or_fix(text: str, config: RunnableConfig):
    fixing_chain = (
        ChatPromptTemplate.from_template(
            "Fix the following text:\n\n```text\n{input}\n```\nError: {error}"
            " Don't narrate, just respond with the fixed data."
        )
        | ChatOpenAI()
        | StrOutputParser()
    )
    for _ in range(3):
        try:
            return json.loads(text)
        except Exception as e:
            text = fixing_chain.invoke({"input": text, "error": e}, config)
    return "Failed to parse"

```


```
from langchain_community.callbacks import get_openai_callback

with get_openai_callback() as cb:
    output = RunnableLambda(parse_or_fix).invoke(
        "{foo: bar}", {"tags": ["my-tag"], "callbacks": [cb]}
    )
    print(output)
    print(cb)

```


```
{'foo': 'bar'}
Tokens Used: 62
    Prompt Tokens: 56
    Completion Tokens: 6
Successful Requests: 1
Total Cost (USD): $9.6e-05

```


Streaming
---------

You can use generator functions (ie. functions that use the `yield` keyword, and behave like iterators) in a LCEL pipeline.

The signature of these generators should be `Iterator[Input] -> Iterator[Output]`. Or for async generators: `AsyncIterator[Input] -> AsyncIterator[Output]`.

These are useful for: - implementing a custom output parser - modifying the output of a previous step, while preserving streaming capabilities

Here’s an example of a custom output parser for comma-separated lists:

```
from typing import Iterator, List

prompt = ChatPromptTemplate.from_template(
    "Write a comma-separated list of 5 animals similar to: {animal}. Do not include numbers"
)
model = ChatOpenAI(temperature=0.0)

str_chain = prompt | model | StrOutputParser()

```


```
for chunk in str_chain.stream({"animal": "bear"}):
    print(chunk, end="", flush=True)

```


```
lion, tiger, wolf, gorilla, panda

```


```
str_chain.invoke({"animal": "bear"})

```


```
'lion, tiger, wolf, gorilla, panda'

```


```
# This is a custom parser that splits an iterator of llm tokens
# into a list of strings separated by commas
def split_into_list(input: Iterator[str]) -> Iterator[List[str]]:
    # hold partial input until we get a comma
    buffer = ""
    for chunk in input:
        # add current chunk to buffer
        buffer += chunk
        # while there are commas in the buffer
        while "," in buffer:
            # split buffer on comma
            comma_index = buffer.index(",")
            # yield everything before the comma
            yield [buffer[:comma_index].strip()]
            # save the rest for the next iteration
            buffer = buffer[comma_index + 1 :]
    # yield the last chunk
    yield [buffer.strip()]

```


```
list_chain = str_chain | split_into_list

```


```
for chunk in list_chain.stream({"animal": "bear"}):
    print(chunk, flush=True)

```


```
['lion']
['tiger']
['wolf']
['gorilla']
['panda']

```


```
list_chain.invoke({"animal": "bear"})

```


```
['lion', 'tiger', 'wolf', 'gorilla', 'elephant']

```


Async version[​](#async-version "Direct link to Async version")
---------------------------------------------------------------

```
from typing import AsyncIterator


async def asplit_into_list(
    input: AsyncIterator[str],
) -> AsyncIterator[List[str]]:  # async def
    buffer = ""
    async for (
        chunk
    ) in input:  # `input` is a `async_generator` object, so use `async for`
        buffer += chunk
        while "," in buffer:
            comma_index = buffer.index(",")
            yield [buffer[:comma_index].strip()]
            buffer = buffer[comma_index + 1 :]
    yield [buffer.strip()]


list_chain = str_chain | asplit_into_list

```


```
async for chunk in list_chain.astream({"animal": "bear"}):
    print(chunk, flush=True)

```


```
['lion']
['tiger']
['wolf']
['gorilla']
['panda']

```


```
await list_chain.ainvoke({"animal": "bear"})

```


```
['lion', 'tiger', 'wolf', 'gorilla', 'panda']

```
