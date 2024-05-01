# Streaming | 🦜️🔗 LangChain
Streaming is critical in making applications based on LLMs feel responsive to end-users.

Important LangChain primitives like LLMs, parsers, prompts, retrievers, and agents implement the LangChain [Runnable Interface](https://python.langchain.com/docs/expression_language/interface/).

This interface provides two general approaches to stream content:

1.  sync `stream` and async `astream`: a **default implementation** of streaming that streams the **final output** from the chain.
2.  async `astream_events` and async `astream_log`: these provide a way to stream both **intermediate steps** and **final output** from the chain.

Let’s take a look at both approaches, and try to understand how to use them. 🥷

Using Stream[​](#using-stream "Direct link to Using Stream")
------------------------------------------------------------

All `Runnable` objects implement a sync method called `stream` and an async variant called `astream`.

These methods are designed to stream the final output in chunks, yielding each chunk as soon as it is available.

Streaming is only possible if all steps in the program know how to process an **input stream**; i.e., process an input chunk one at a time, and yield a corresponding output chunk.

The complexity of this processing can vary, from straightforward tasks like emitting tokens produced by an LLM, to more challenging ones like streaming parts of JSON results before the entire JSON is complete.

The best place to start exploring streaming is with the single most important components in LLMs apps– the LLMs themselves!

### LLMs and Chat Models[​](#llms-and-chat-models "Direct link to LLMs and Chat Models")

Large language models and their chat variants are the primary bottleneck in LLM based apps. 🙊

Large language models can take **several seconds** to generate a complete response to a query. This is far slower than the **~200-300 ms** threshold at which an application feels responsive to an end user.

The key strategy to make the application feel more responsive is to show intermediate progress; viz., to stream the output from the model **token by token**.

We will show examples of streaming using the chat model from [Anthropic](https://python.langchain.com/docs/integrations/platforms/anthropic/). To use the model, you will need to install the `langchain-anthropic` package. You can do this with the following command:

```
pip install -qU langchain-anthropic

```


```
# Showing the example using anthropic, but you can use
# your favorite chat model!
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic()

chunks = []
async for chunk in model.astream("hello. tell me something about yourself"):
    chunks.append(chunk)
    print(chunk.content, end="|", flush=True)

```


```
 Hello|!| My| name| is| Claude|.| I|'m| an| AI| assistant| created| by| An|throp|ic| to| be| helpful|,| harmless|,| and| honest|.||

```


Let’s inspect one of the chunks

```
AIMessageChunk(content=' Hello')

```


We got back something called an `AIMessageChunk`. This chunk represents a part of an `AIMessage`.

Message chunks are additive by design – one can simply add them up to get the state of the response so far!

```
chunks[0] + chunks[1] + chunks[2] + chunks[3] + chunks[4]

```


```
AIMessageChunk(content=' Hello! My name is')

```


### Chains[​](#chains "Direct link to Chains")

Virtually all LLM applications involve more steps than just a call to a language model.

Let’s build a simple chain using `LangChain Expression Language` (`LCEL`) that combines a prompt, model and a parser and verify that streaming works.

We will use `StrOutputParser` to parse the output from the model. This is a simple parser that extracts the `content` field from an `AIMessageChunk`, giving us the `token` returned by the model.

tip

LCEL is a _declarative_ way to specify a “program” by chainining together different LangChain primitives. Chains created using LCEL benefit from an automatic implementation of `stream` and `astream` allowing streaming of the final output. In fact, chains created with LCEL implement the entire standard Runnable interface.

```
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
parser = StrOutputParser()
chain = prompt | model | parser

async for chunk in chain.astream({"topic": "parrot"}):
    print(chunk, end="|", flush=True)

```


```
 Here|'s| a| silly| joke| about| a| par|rot|:|

What| kind| of| teacher| gives| good| advice|?| An| ap|-|parent| (|app|arent|)| one|!||

```


You might notice above that `parser` actually doesn’t block the streaming output from the model, and instead processes each chunk individually. Many of the [LCEL primitives](https://python.langchain.com/docs/expression_language/primitives/) also support this kind of transform-style passthrough streaming, which can be very convenient when constructing apps.

Certain runnables, like [prompt templates](https://python.langchain.com/docs/modules/model_io/prompts/) and [chat models](https://python.langchain.com/docs/modules/model_io/chat/), cannot process individual chunks and instead aggregate all previous steps. This will interrupt the streaming process. Custom functions can be [designed to return generators](https://python.langchain.com/docs/expression_language/primitives/functions/#streaming), which

note

If the above functionality is not relevant to what you’re building, you do not have to use the `LangChain Expression Language` to use LangChain and can instead rely on a standard **imperative** programming approach by caling `invoke`, `batch` or `stream` on each component individually, assigning the results to variables and then using them downstream as you see fit.

If that works for your needs, then that’s fine by us 👌!

### Working with Input Streams[​](#working-with-input-streams "Direct link to Working with Input Streams")

What if you wanted to stream JSON from the output as it was being generated?

If you were to rely on `json.loads` to parse the partial json, the parsing would fail as the partial json wouldn’t be valid json.

You’d likely be at a complete loss of what to do and claim that it wasn’t possible to stream JSON.

Well, turns out there is a way to do it – the parser needs to operate on the **input stream**, and attempt to “auto-complete” the partial json into a valid state.

Let’s see such a parser in action to understand what this means.

```
from langchain_core.output_parsers import JsonOutputParser

chain = (
    model | JsonOutputParser()
)  # Due to a bug in older versions of Langchain, JsonOutputParser did not stream results from some models
async for text in chain.astream(
    'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`'
):
    print(text, flush=True)

```


```
{}
{'countries': []}
{'countries': [{}]}
{'countries': [{'name': ''}]}
{'countries': [{'name': 'France'}]}
{'countries': [{'name': 'France', 'population': 67}]}
{'countries': [{'name': 'France', 'population': 6739}]}
{'countries': [{'name': 'France', 'population': 673915}]}
{'countries': [{'name': 'France', 'population': 67391582}]}
{'countries': [{'name': 'France', 'population': 67391582}, {}]}
{'countries': [{'name': 'France', 'population': 67391582}, {'name': ''}]}
{'countries': [{'name': 'France', 'population': 67391582}, {'name': 'Sp'}]}
{'countries': [{'name': 'France', 'population': 67391582}, {'name': 'Spain'}]}
{'countries': [{'name': 'France', 'population': 67391582}, {'name': 'Spain', 'population': 46}]}
{'countries': [{'name': 'France', 'population': 67391582}, {'name': 'Spain', 'population': 4675}]}
{'countries': [{'name': 'France', 'population': 67391582}, {'name': 'Spain', 'population': 467547}]}
{'countries': [{'name': 'France', 'population': 67391582}, {'name': 'Spain', 'population': 46754778}]}
{'countries': [{'name': 'France', 'population': 67391582}, {'name': 'Spain', 'population': 46754778}, {}]}
{'countries': [{'name': 'France', 'population': 67391582}, {'name': 'Spain', 'population': 46754778}, {'name': ''}]}
{'countries': [{'name': 'France', 'population': 67391582}, {'name': 'Spain', 'population': 46754778}, {'name': 'Japan'}]}
{'countries': [{'name': 'France', 'population': 67391582}, {'name': 'Spain', 'population': 46754778}, {'name': 'Japan', 'population': 12}]}
{'countries': [{'name': 'France', 'population': 67391582}, {'name': 'Spain', 'population': 46754778}, {'name': 'Japan', 'population': 12647}]}
{'countries': [{'name': 'France', 'population': 67391582}, {'name': 'Spain', 'population': 46754778}, {'name': 'Japan', 'population': 1264764}]}
{'countries': [{'name': 'France', 'population': 67391582}, {'name': 'Spain', 'population': 46754778}, {'name': 'Japan', 'population': 126476461}]}

```


Now, let’s **break** streaming. We’ll use the previous example and append an extraction function at the end that extracts the country names from the finalized JSON.

danger

Any steps in the chain that operate on **finalized inputs** rather than on **input streams** can break streaming functionality via `stream` or `astream`.

tip

Later, we will discuss the `astream_events` API which streams results from intermediate steps. This API will stream results from intermediate steps even if the chain contains steps that only operate on **finalized inputs**.

```
from langchain_core.output_parsers import (
    JsonOutputParser,
)


# A function that operates on finalized inputs
# rather than on an input_stream
def _extract_country_names(inputs):
    """A function that does not operates on input streams and breaks streaming."""
    if not isinstance(inputs, dict):
        return ""

    if "countries" not in inputs:
        return ""

    countries = inputs["countries"]

    if not isinstance(countries, list):
        return ""

    country_names = [
        country.get("name") for country in countries if isinstance(country, dict)
    ]
    return country_names


chain = model | JsonOutputParser() | _extract_country_names

async for text in chain.astream(
    'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`'
):
    print(text, end="|", flush=True)

```


```
['France', 'Spain', 'Japan']|

```


#### Generator Functions[​](#generator-functions "Direct link to Generator Functions")

Le’ts fix the streaming using a generator function that can operate on the **input stream**.

tip

A generator function (a function that uses `yield`) allows writing code that operators on **input streams**

```
from langchain_core.output_parsers import JsonOutputParser


async def _extract_country_names_streaming(input_stream):
    """A function that operates on input streams."""
    country_names_so_far = set()

    async for input in input_stream:
        if not isinstance(input, dict):
            continue

        if "countries" not in input:
            continue

        countries = input["countries"]

        if not isinstance(countries, list):
            continue

        for country in countries:
            name = country.get("name")
            if not name:
                continue
            if name not in country_names_so_far:
                yield name
                country_names_so_far.add(name)


chain = model | JsonOutputParser() | _extract_country_names_streaming

async for text in chain.astream(
    'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`'
):
    print(text, end="|", flush=True)

```


note

Because the code above is relying on JSON auto-completion, you may see partial names of countries (e.g., `Sp` and `Spain`), which is not what one would want for an extraction result!

We’re focusing on streaming concepts, not necessarily the results of the chains.

### Non-streaming components[​](#non-streaming-components "Direct link to Non-streaming components")

Some built-in components like Retrievers do not offer any `streaming`. What happens if we try to `stream` them? 🤨

```
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

vectorstore = FAISS.from_texts(
    ["harrison worked at kensho", "harrison likes spicy food"],
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

chunks = [chunk for chunk in retriever.stream("where did harrison work?")]
chunks

```


```
[[Document(page_content='harrison worked at kensho'),
  Document(page_content='harrison likes spicy food')]]

```


Stream just yielded the final result from that component.

This is OK 🥹! Not all components have to implement streaming – in some cases streaming is either unnecessary, difficult or just doesn’t make sense.

tip

An LCEL chain constructed using non-streaming components, will still be able to stream in a lot of cases, with streaming of partial output starting after the last non-streaming step in the chain.

```
retrieval_chain = (
    {
        "context": retriever.with_config(run_name="Docs"),
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

```


```
for chunk in retrieval_chain.stream(
    "Where did harrison work? " "Write 3 made up sentences about this place."
):
    print(chunk, end="|", flush=True)

```


```
 Based| on| the| given| context|,| the| only| information| provided| about| where| Harrison| worked| is| that| he| worked| at| Ken|sh|o|.| Since| there| are| no| other| details| provided| about| Ken|sh|o|,| I| do| not| have| enough| information| to| write| 3| additional| made| up| sentences| about| this| place|.| I| can| only| state| that| Harrison| worked| at| Ken|sh|o|.||

```


Now that we’ve seen how `stream` and `astream` work, let’s venture into the world of streaming events. 🏞️

Using Stream Events[​](#using-stream-events "Direct link to Using Stream Events")
---------------------------------------------------------------------------------

Event Streaming is a **beta** API. This API may change a bit based on feedback.

note

Introduced in langchain-core **0.1.14**.

```
import langchain_core

langchain_core.__version__

```


For the `astream_events` API to work properly:

*   Use `async` throughout the code to the extent possible (e.g., async tools etc)
*   Propagate callbacks if defining custom functions / runnables
*   Whenever using runnables without LCEL, make sure to call `.astream()` on LLMs rather than `.ainvoke` to force the LLM to stream tokens.
*   Let us know if anything doesn’t work as expected! :)

### Event Reference[​](#event-reference "Direct link to Event Reference")

Below is a reference table that shows some events that might be emitted by the various Runnable objects.

note

When streaming is implemented properly, the inputs to a runnable will not be known until after the input stream has been entirely consumed. This means that `inputs` will often be included only for `end` events and rather than for `start` events.



* event: on_chat_model_start
  * name: [model name]
  * chunk: 
  * input: {“messages”: [[SystemMessage, HumanMessage]]}
  * output: 
* event: on_chat_model_stream
  * name: [model name]
  * chunk: AIMessageChunk(content=“hello”)
  * input: 
  * output: 
* event: on_chat_model_end
  * name: [model name]
  * chunk: 
  * input: {“messages”: [[SystemMessage, HumanMessage]]}
  * output: {“generations”: […], “llm_output”: None, …}
* event: on_llm_start
  * name: [model name]
  * chunk: 
  * input: {‘input’: ‘hello’}
  * output: 
* event: on_llm_stream
  * name: [model name]
  * chunk: ‘Hello’
  * input: 
  * output: 
* event: on_llm_end
  * name: [model name]
  * chunk: 
  * input: ‘Hello human!’
  * output: 
* event: on_chain_start
  * name: format_docs
  * chunk: 
  * input: 
  * output: 
* event: on_chain_stream
  * name: format_docs
  * chunk: “hello world!, goodbye world!”
  * input: 
  * output: 
* event: on_chain_end
  * name: format_docs
  * chunk: 
  * input: [Document(…)]
  * output: “hello world!, goodbye world!”
* event: on_tool_start
  * name: some_tool
  * chunk: 
  * input: {“x”: 1, “y”: “2”}
  * output: 
* event: on_tool_stream
  * name: some_tool
  * chunk: {“x”: 1, “y”: “2”}
  * input: 
  * output: 
* event: on_tool_end
  * name: some_tool
  * chunk: 
  * input: 
  * output: {“x”: 1, “y”: “2”}
* event: on_retriever_start
  * name: [retriever name]
  * chunk: 
  * input: {“query”: “hello”}
  * output: 
* event: on_retriever_chunk
  * name: [retriever name]
  * chunk: {documents: […]}
  * input: 
  * output: 
* event: on_retriever_end
  * name: [retriever name]
  * chunk: 
  * input: {“query”: “hello”}
  * output: {documents: […]}
* event: on_prompt_start
  * name: [template_name]
  * chunk: 
  * input: {“question”: “hello”}
  * output: 
* event: on_prompt_end
  * name: [template_name]
  * chunk: 
  * input: {“question”: “hello”}
  * output: ChatPromptValue(messages: [SystemMessage, …])


### Chat Model[​](#chat-model "Direct link to Chat Model")

Let’s start off by looking at the events produced by a chat model.

```
events = []
async for event in model.astream_events("hello", version="v1"):
    events.append(event)

```


```
/home/eugene/src/langchain/libs/core/langchain_core/_api/beta_decorator.py:86: LangChainBetaWarning: This API is in beta and may change in the future.
  warn_beta(

```


note

Hey what’s that funny version=“v1” parameter in the API?! 😾

This is a **beta API**, and we’re almost certainly going to make some changes to it.

This version parameter will allow us to minimize such breaking changes to your code.

In short, we are annoying you now, so we don’t have to annoy you later.

Let’s take a look at the few of the start event and a few of the end events.

```
[{'event': 'on_chat_model_start',
  'run_id': '555843ed-3d24-4774-af25-fbf030d5e8c4',
  'name': 'ChatAnthropic',
  'tags': [],
  'metadata': {},
  'data': {'input': 'hello'}},
 {'event': 'on_chat_model_stream',
  'run_id': '555843ed-3d24-4774-af25-fbf030d5e8c4',
  'tags': [],
  'metadata': {},
  'name': 'ChatAnthropic',
  'data': {'chunk': AIMessageChunk(content=' Hello')}},
 {'event': 'on_chat_model_stream',
  'run_id': '555843ed-3d24-4774-af25-fbf030d5e8c4',
  'tags': [],
  'metadata': {},
  'name': 'ChatAnthropic',
  'data': {'chunk': AIMessageChunk(content='!')}}]

```


```
[{'event': 'on_chat_model_stream',
  'run_id': '555843ed-3d24-4774-af25-fbf030d5e8c4',
  'tags': [],
  'metadata': {},
  'name': 'ChatAnthropic',
  'data': {'chunk': AIMessageChunk(content='')}},
 {'event': 'on_chat_model_end',
  'name': 'ChatAnthropic',
  'run_id': '555843ed-3d24-4774-af25-fbf030d5e8c4',
  'tags': [],
  'metadata': {},
  'data': {'output': AIMessageChunk(content=' Hello!')}}]

```


### Chain[​](#chain "Direct link to Chain")

Let’s revisit the example chain that parsed streaming JSON to explore the streaming events API.

```
chain = (
    model | JsonOutputParser()
)  # Due to a bug in older versions of Langchain, JsonOutputParser did not stream results from some models

events = [
    event
    async for event in chain.astream_events(
        'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`',
        version="v1",
    )
]

```


If you examine at the first few events, you’ll notice that there are **3** different start events rather than **2** start events.

The three start events correspond to:

1.  The chain (model + parser)
2.  The model
3.  The parser

```
[{'event': 'on_chain_start',
  'run_id': 'b1074bff-2a17-458b-9e7b-625211710df4',
  'name': 'RunnableSequence',
  'tags': [],
  'metadata': {},
  'data': {'input': 'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`'}},
 {'event': 'on_chat_model_start',
  'name': 'ChatAnthropic',
  'run_id': '6072be59-1f43-4f1c-9470-3b92e8406a99',
  'tags': ['seq:step:1'],
  'metadata': {},
  'data': {'input': {'messages': [[HumanMessage(content='output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`')]]}}},
 {'event': 'on_parser_start',
  'name': 'JsonOutputParser',
  'run_id': 'bf978194-0eda-4494-ad15-3a5bfe69cd59',
  'tags': ['seq:step:2'],
  'metadata': {},
  'data': {}}]

```


What do you think you’d see if you looked at the last 3 events? what about the middle?

Let’s use this API to take output the stream events from the model and the parser. We’re ignoring start events, end events and events from the chain.

```
num_events = 0

async for event in chain.astream_events(
    'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`',
    version="v1",
):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        print(
            f"Chat model chunk: {repr(event['data']['chunk'].content)}",
            flush=True,
        )
    if kind == "on_parser_stream":
        print(f"Parser chunk: {event['data']['chunk']}", flush=True)
    num_events += 1
    if num_events > 30:
        # Truncate the output
        print("...")
        break

```


```
Chat model chunk: ' Here'
Chat model chunk: ' is'
Chat model chunk: ' the'
Chat model chunk: ' JSON'
Chat model chunk: ' with'
Chat model chunk: ' the'
Chat model chunk: ' requested'
Chat model chunk: ' countries'
Chat model chunk: ' and'
Chat model chunk: ' their'
Chat model chunk: ' populations'
Chat model chunk: ':'
Chat model chunk: '\n\n```'
Chat model chunk: 'json'
Parser chunk: {}
Chat model chunk: '\n{'
Chat model chunk: '\n '
Chat model chunk: ' "'
Chat model chunk: 'countries'
Chat model chunk: '":'
Parser chunk: {'countries': []}
Chat model chunk: ' ['
Chat model chunk: '\n   '
Parser chunk: {'countries': [{}]}
Chat model chunk: ' {'
...

```


Because both the model and the parser support streaming, we see sreaming events from both components in real time! Kind of cool isn’t it? 🦜

### Filtering Events[​](#filtering-events "Direct link to Filtering Events")

Because this API produces so many events, it is useful to be able to filter on events.

You can filter by either component `name`, component `tags` or component `type`.

#### By Name[​](#by-name "Direct link to By Name")

```
chain = model.with_config({"run_name": "model"}) | JsonOutputParser().with_config(
    {"run_name": "my_parser"}
)

max_events = 0
async for event in chain.astream_events(
    'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`',
    version="v1",
    include_names=["my_parser"],
):
    print(event)
    max_events += 1
    if max_events > 10:
        # Truncate output
        print("...")
        break

```


```
{'event': 'on_parser_start', 'name': 'my_parser', 'run_id': 'f2ac1d1c-e14a-45fc-8990-e5c24e707299', 'tags': ['seq:step:2'], 'metadata': {}, 'data': {}}
{'event': 'on_parser_stream', 'name': 'my_parser', 'run_id': 'f2ac1d1c-e14a-45fc-8990-e5c24e707299', 'tags': ['seq:step:2'], 'metadata': {}, 'data': {'chunk': {}}}
{'event': 'on_parser_stream', 'name': 'my_parser', 'run_id': 'f2ac1d1c-e14a-45fc-8990-e5c24e707299', 'tags': ['seq:step:2'], 'metadata': {}, 'data': {'chunk': {'countries': []}}}
{'event': 'on_parser_stream', 'name': 'my_parser', 'run_id': 'f2ac1d1c-e14a-45fc-8990-e5c24e707299', 'tags': ['seq:step:2'], 'metadata': {}, 'data': {'chunk': {'countries': [{}]}}}
{'event': 'on_parser_stream', 'name': 'my_parser', 'run_id': 'f2ac1d1c-e14a-45fc-8990-e5c24e707299', 'tags': ['seq:step:2'], 'metadata': {}, 'data': {'chunk': {'countries': [{'name': ''}]}}}
{'event': 'on_parser_stream', 'name': 'my_parser', 'run_id': 'f2ac1d1c-e14a-45fc-8990-e5c24e707299', 'tags': ['seq:step:2'], 'metadata': {}, 'data': {'chunk': {'countries': [{'name': 'France'}]}}}
{'event': 'on_parser_stream', 'name': 'my_parser', 'run_id': 'f2ac1d1c-e14a-45fc-8990-e5c24e707299', 'tags': ['seq:step:2'], 'metadata': {}, 'data': {'chunk': {'countries': [{'name': 'France', 'population': 67}]}}}
{'event': 'on_parser_stream', 'name': 'my_parser', 'run_id': 'f2ac1d1c-e14a-45fc-8990-e5c24e707299', 'tags': ['seq:step:2'], 'metadata': {}, 'data': {'chunk': {'countries': [{'name': 'France', 'population': 6739}]}}}
{'event': 'on_parser_stream', 'name': 'my_parser', 'run_id': 'f2ac1d1c-e14a-45fc-8990-e5c24e707299', 'tags': ['seq:step:2'], 'metadata': {}, 'data': {'chunk': {'countries': [{'name': 'France', 'population': 673915}]}}}
{'event': 'on_parser_stream', 'name': 'my_parser', 'run_id': 'f2ac1d1c-e14a-45fc-8990-e5c24e707299', 'tags': ['seq:step:2'], 'metadata': {}, 'data': {'chunk': {'countries': [{'name': 'France', 'population': 67391582}]}}}
{'event': 'on_parser_stream', 'name': 'my_parser', 'run_id': 'f2ac1d1c-e14a-45fc-8990-e5c24e707299', 'tags': ['seq:step:2'], 'metadata': {}, 'data': {'chunk': {'countries': [{'name': 'France', 'population': 67391582}, {}]}}}
...

```


#### By Type[​](#by-type "Direct link to By Type")

```
chain = model.with_config({"run_name": "model"}) | JsonOutputParser().with_config(
    {"run_name": "my_parser"}
)

max_events = 0
async for event in chain.astream_events(
    'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`',
    version="v1",
    include_types=["chat_model"],
):
    print(event)
    max_events += 1
    if max_events > 10:
        # Truncate output
        print("...")
        break

```


```
{'event': 'on_chat_model_start', 'name': 'model', 'run_id': '98a6e192-8159-460c-ba73-6dfc921e3777', 'tags': ['seq:step:1'], 'metadata': {}, 'data': {'input': {'messages': [[HumanMessage(content='output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`')]]}}}
{'event': 'on_chat_model_stream', 'name': 'model', 'run_id': '98a6e192-8159-460c-ba73-6dfc921e3777', 'tags': ['seq:step:1'], 'metadata': {}, 'data': {'chunk': AIMessageChunk(content=' Here')}}
{'event': 'on_chat_model_stream', 'name': 'model', 'run_id': '98a6e192-8159-460c-ba73-6dfc921e3777', 'tags': ['seq:step:1'], 'metadata': {}, 'data': {'chunk': AIMessageChunk(content=' is')}}
{'event': 'on_chat_model_stream', 'name': 'model', 'run_id': '98a6e192-8159-460c-ba73-6dfc921e3777', 'tags': ['seq:step:1'], 'metadata': {}, 'data': {'chunk': AIMessageChunk(content=' the')}}
{'event': 'on_chat_model_stream', 'name': 'model', 'run_id': '98a6e192-8159-460c-ba73-6dfc921e3777', 'tags': ['seq:step:1'], 'metadata': {}, 'data': {'chunk': AIMessageChunk(content=' JSON')}}
{'event': 'on_chat_model_stream', 'name': 'model', 'run_id': '98a6e192-8159-460c-ba73-6dfc921e3777', 'tags': ['seq:step:1'], 'metadata': {}, 'data': {'chunk': AIMessageChunk(content=' with')}}
{'event': 'on_chat_model_stream', 'name': 'model', 'run_id': '98a6e192-8159-460c-ba73-6dfc921e3777', 'tags': ['seq:step:1'], 'metadata': {}, 'data': {'chunk': AIMessageChunk(content=' the')}}
{'event': 'on_chat_model_stream', 'name': 'model', 'run_id': '98a6e192-8159-460c-ba73-6dfc921e3777', 'tags': ['seq:step:1'], 'metadata': {}, 'data': {'chunk': AIMessageChunk(content=' requested')}}
{'event': 'on_chat_model_stream', 'name': 'model', 'run_id': '98a6e192-8159-460c-ba73-6dfc921e3777', 'tags': ['seq:step:1'], 'metadata': {}, 'data': {'chunk': AIMessageChunk(content=' countries')}}
{'event': 'on_chat_model_stream', 'name': 'model', 'run_id': '98a6e192-8159-460c-ba73-6dfc921e3777', 'tags': ['seq:step:1'], 'metadata': {}, 'data': {'chunk': AIMessageChunk(content=' and')}}
{'event': 'on_chat_model_stream', 'name': 'model', 'run_id': '98a6e192-8159-460c-ba73-6dfc921e3777', 'tags': ['seq:step:1'], 'metadata': {}, 'data': {'chunk': AIMessageChunk(content=' their')}}
...

```


#### By Tags[​](#by-tags "Direct link to By Tags")

caution

Tags are inherited by child components of a given runnable.

If you’re using tags to filter, make sure that this is what you want.

```
chain = (model | JsonOutputParser()).with_config({"tags": ["my_chain"]})

max_events = 0
async for event in chain.astream_events(
    'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`',
    version="v1",
    include_tags=["my_chain"],
):
    print(event)
    max_events += 1
    if max_events > 10:
        # Truncate output
        print("...")
        break

```


```
{'event': 'on_chain_start', 'run_id': '190875f3-3fb7-49ad-9b6e-f49da22f3e49', 'name': 'RunnableSequence', 'tags': ['my_chain'], 'metadata': {}, 'data': {'input': 'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`'}}
{'event': 'on_chat_model_start', 'name': 'ChatAnthropic', 'run_id': 'ff58f732-b494-4ff9-852a-783d42f4455d', 'tags': ['seq:step:1', 'my_chain'], 'metadata': {}, 'data': {'input': {'messages': [[HumanMessage(content='output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`')]]}}}
{'event': 'on_parser_start', 'name': 'JsonOutputParser', 'run_id': '3b5e4ca1-40fe-4a02-9a19-ba2a43a6115c', 'tags': ['seq:step:2', 'my_chain'], 'metadata': {}, 'data': {}}
{'event': 'on_chat_model_stream', 'name': 'ChatAnthropic', 'run_id': 'ff58f732-b494-4ff9-852a-783d42f4455d', 'tags': ['seq:step:1', 'my_chain'], 'metadata': {}, 'data': {'chunk': AIMessageChunk(content=' Here')}}
{'event': 'on_chat_model_stream', 'name': 'ChatAnthropic', 'run_id': 'ff58f732-b494-4ff9-852a-783d42f4455d', 'tags': ['seq:step:1', 'my_chain'], 'metadata': {}, 'data': {'chunk': AIMessageChunk(content=' is')}}
{'event': 'on_chat_model_stream', 'name': 'ChatAnthropic', 'run_id': 'ff58f732-b494-4ff9-852a-783d42f4455d', 'tags': ['seq:step:1', 'my_chain'], 'metadata': {}, 'data': {'chunk': AIMessageChunk(content=' the')}}
{'event': 'on_chat_model_stream', 'name': 'ChatAnthropic', 'run_id': 'ff58f732-b494-4ff9-852a-783d42f4455d', 'tags': ['seq:step:1', 'my_chain'], 'metadata': {}, 'data': {'chunk': AIMessageChunk(content=' JSON')}}
{'event': 'on_chat_model_stream', 'name': 'ChatAnthropic', 'run_id': 'ff58f732-b494-4ff9-852a-783d42f4455d', 'tags': ['seq:step:1', 'my_chain'], 'metadata': {}, 'data': {'chunk': AIMessageChunk(content=' with')}}
{'event': 'on_chat_model_stream', 'name': 'ChatAnthropic', 'run_id': 'ff58f732-b494-4ff9-852a-783d42f4455d', 'tags': ['seq:step:1', 'my_chain'], 'metadata': {}, 'data': {'chunk': AIMessageChunk(content=' the')}}
{'event': 'on_chat_model_stream', 'name': 'ChatAnthropic', 'run_id': 'ff58f732-b494-4ff9-852a-783d42f4455d', 'tags': ['seq:step:1', 'my_chain'], 'metadata': {}, 'data': {'chunk': AIMessageChunk(content=' requested')}}
{'event': 'on_chat_model_stream', 'name': 'ChatAnthropic', 'run_id': 'ff58f732-b494-4ff9-852a-783d42f4455d', 'tags': ['seq:step:1', 'my_chain'], 'metadata': {}, 'data': {'chunk': AIMessageChunk(content=' countries')}}
...

```


### Non-streaming components[​](#non-streaming-components-1 "Direct link to Non-streaming components")

Remember how some components don’t stream well because they don’t operate on **input streams**?

While such components can break streaming of the final output when using `astream`, `astream_events` will still yield streaming events from intermediate steps that support streaming!

```
# Function that does not support streaming.
# It operates on the finalizes inputs rather than
# operating on the input stream.
def _extract_country_names(inputs):
    """A function that does not operates on input streams and breaks streaming."""
    if not isinstance(inputs, dict):
        return ""

    if "countries" not in inputs:
        return ""

    countries = inputs["countries"]

    if not isinstance(countries, list):
        return ""

    country_names = [
        country.get("name") for country in countries if isinstance(country, dict)
    ]
    return country_names


chain = (
    model | JsonOutputParser() | _extract_country_names
)  # This parser only works with OpenAI right now

```


As expected, the `astream` API doesn’t work correctly because `_extract_country_names` doesn’t operate on streams.

```
async for chunk in chain.astream(
    'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`',
):
    print(chunk, flush=True)

```


```
['France', 'Spain', 'Japan']

```


Now, let’s confirm that with astream\_events we’re still seeing streaming output from the model and the parser.

```
num_events = 0

async for event in chain.astream_events(
    'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`',
    version="v1",
):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        print(
            f"Chat model chunk: {repr(event['data']['chunk'].content)}",
            flush=True,
        )
    if kind == "on_parser_stream":
        print(f"Parser chunk: {event['data']['chunk']}", flush=True)
    num_events += 1
    if num_events > 30:
        # Truncate the output
        print("...")
        break

```


```
Chat model chunk: ' Here'
Chat model chunk: ' is'
Chat model chunk: ' the'
Chat model chunk: ' JSON'
Chat model chunk: ' with'
Chat model chunk: ' the'
Chat model chunk: ' requested'
Chat model chunk: ' countries'
Chat model chunk: ' and'
Chat model chunk: ' their'
Chat model chunk: ' populations'
Chat model chunk: ':'
Chat model chunk: '\n\n```'
Chat model chunk: 'json'
Parser chunk: {}
Chat model chunk: '\n{'
Chat model chunk: '\n '
Chat model chunk: ' "'
Chat model chunk: 'countries'
Chat model chunk: '":'
Parser chunk: {'countries': []}
Chat model chunk: ' ['
Chat model chunk: '\n   '
Parser chunk: {'countries': [{}]}
Chat model chunk: ' {'
Chat model chunk: '\n     '
Chat model chunk: ' "'
...

```


### Propagating Callbacks[​](#propagating-callbacks "Direct link to Propagating Callbacks")

caution

If you’re using invoking runnables inside your tools, you need to propagate callbacks to the runnable; otherwise, no stream events will be generated.

note

When using RunnableLambdas or @chain decorator, callbacks are propagated automatically behind the scenes.

```
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool


def reverse_word(word: str):
    return word[::-1]


reverse_word = RunnableLambda(reverse_word)


@tool
def bad_tool(word: str):
    """Custom tool that doesn't propagate callbacks."""
    return reverse_word.invoke(word)


async for event in bad_tool.astream_events("hello", version="v1"):
    print(event)

```


```
{'event': 'on_tool_start', 'run_id': 'ae7690f8-ebc9-4886-9bbe-cb336ff274f2', 'name': 'bad_tool', 'tags': [], 'metadata': {}, 'data': {'input': 'hello'}}
{'event': 'on_tool_stream', 'run_id': 'ae7690f8-ebc9-4886-9bbe-cb336ff274f2', 'tags': [], 'metadata': {}, 'name': 'bad_tool', 'data': {'chunk': 'olleh'}}
{'event': 'on_tool_end', 'name': 'bad_tool', 'run_id': 'ae7690f8-ebc9-4886-9bbe-cb336ff274f2', 'tags': [], 'metadata': {}, 'data': {'output': 'olleh'}}

```


Here’s a re-implementation that does propagate callbacks correctly. You’ll notice that now we’re getting events from the `reverse_word` runnable as well.

```
@tool
def correct_tool(word: str, callbacks):
    """A tool that correctly propagates callbacks."""
    return reverse_word.invoke(word, {"callbacks": callbacks})


async for event in correct_tool.astream_events("hello", version="v1"):
    print(event)

```


```
{'event': 'on_tool_start', 'run_id': '384f1710-612e-4022-a6d4-8a7bb0cc757e', 'name': 'correct_tool', 'tags': [], 'metadata': {}, 'data': {'input': 'hello'}}
{'event': 'on_chain_start', 'name': 'reverse_word', 'run_id': 'c4882303-8867-4dff-b031-7d9499b39dda', 'tags': [], 'metadata': {}, 'data': {'input': 'hello'}}
{'event': 'on_chain_end', 'name': 'reverse_word', 'run_id': 'c4882303-8867-4dff-b031-7d9499b39dda', 'tags': [], 'metadata': {}, 'data': {'input': 'hello', 'output': 'olleh'}}
{'event': 'on_tool_stream', 'run_id': '384f1710-612e-4022-a6d4-8a7bb0cc757e', 'tags': [], 'metadata': {}, 'name': 'correct_tool', 'data': {'chunk': 'olleh'}}
{'event': 'on_tool_end', 'name': 'correct_tool', 'run_id': '384f1710-612e-4022-a6d4-8a7bb0cc757e', 'tags': [], 'metadata': {}, 'data': {'output': 'olleh'}}

```


If you’re invoking runnables from within Runnable Lambdas or @chains, then callbacks will be passed automatically on your behalf.

```
from langchain_core.runnables import RunnableLambda


async def reverse_and_double(word: str):
    return await reverse_word.ainvoke(word) * 2


reverse_and_double = RunnableLambda(reverse_and_double)

await reverse_and_double.ainvoke("1234")

async for event in reverse_and_double.astream_events("1234", version="v1"):
    print(event)

```


```
{'event': 'on_chain_start', 'run_id': '4fe56c7b-6982-4999-a42d-79ba56151176', 'name': 'reverse_and_double', 'tags': [], 'metadata': {}, 'data': {'input': '1234'}}
{'event': 'on_chain_start', 'name': 'reverse_word', 'run_id': '335fe781-8944-4464-8d2e-81f61d1f85f5', 'tags': [], 'metadata': {}, 'data': {'input': '1234'}}
{'event': 'on_chain_end', 'name': 'reverse_word', 'run_id': '335fe781-8944-4464-8d2e-81f61d1f85f5', 'tags': [], 'metadata': {}, 'data': {'input': '1234', 'output': '4321'}}
{'event': 'on_chain_stream', 'run_id': '4fe56c7b-6982-4999-a42d-79ba56151176', 'tags': [], 'metadata': {}, 'name': 'reverse_and_double', 'data': {'chunk': '43214321'}}
{'event': 'on_chain_end', 'name': 'reverse_and_double', 'run_id': '4fe56c7b-6982-4999-a42d-79ba56151176', 'tags': [], 'metadata': {}, 'data': {'output': '43214321'}}

```


And with the @chain decorator:

```
from langchain_core.runnables import chain


@chain
async def reverse_and_double(word: str):
    return await reverse_word.ainvoke(word) * 2


await reverse_and_double.ainvoke("1234")

async for event in reverse_and_double.astream_events("1234", version="v1"):
    print(event)

```


```
{'event': 'on_chain_start', 'run_id': '7485eedb-1854-429c-a2f8-03d01452daef', 'name': 'reverse_and_double', 'tags': [], 'metadata': {}, 'data': {'input': '1234'}}
{'event': 'on_chain_start', 'name': 'reverse_word', 'run_id': 'e7cddab2-9b95-4e80-abaf-4b2429117835', 'tags': [], 'metadata': {}, 'data': {'input': '1234'}}
{'event': 'on_chain_end', 'name': 'reverse_word', 'run_id': 'e7cddab2-9b95-4e80-abaf-4b2429117835', 'tags': [], 'metadata': {}, 'data': {'input': '1234', 'output': '4321'}}
{'event': 'on_chain_stream', 'run_id': '7485eedb-1854-429c-a2f8-03d01452daef', 'tags': [], 'metadata': {}, 'name': 'reverse_and_double', 'data': {'chunk': '43214321'}}
{'event': 'on_chain_end', 'name': 'reverse_and_double', 'run_id': '7485eedb-1854-429c-a2f8-03d01452daef', 'tags': [], 'metadata': {}, 'data': {'output': '43214321'}}

```
