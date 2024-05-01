# Decomposition | 🦜️🔗 LangChain
When a user asks a question there is no guarantee that the relevant results can be returned with a single query. Sometimes to answer a question we need to split it into distinct sub-questions, retrieve results for each sub-question, and then answer using the cumulative context.

For example if a user asks: “How is Web Voyager different from reflection agents”, and we have one document that explains Web Voyager and one that explains reflection agents but no document that compares the two, then we’d likely get better results by retrieving for both “What is Web Voyager” and “What are reflection agents” and combining the retrieved documents than by retrieving based on the user question directly.

This process of splitting an input into multiple distinct sub-queries is what we refer to as **query decomposition**. It is also sometimes referred to as sub-query generation. In this guide we’ll walk through an example of how to do decomposition, using our example of a Q&A bot over the LangChain YouTube videos from the [Quickstart](https://python.langchain.com/docs/use_cases/query_analysis/quickstart/).

Setup[​](#setup "Direct link to Setup")
---------------------------------------

#### Install dependencies[​](#install-dependencies "Direct link to Install dependencies")

```
# %pip install -qU langchain langchain-openai

```


#### Set environment variables[​](#set-environment-variables "Direct link to Set environment variables")

We’ll use OpenAI in this example:

```
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

# Optional, uncomment to trace runs with LangSmith. Sign up here: https://smith.langchain.com.
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

```


Query generation[​](#query-generation "Direct link to Query generation")
------------------------------------------------------------------------

To convert user questions to a list of sub questions we’ll use OpenAI’s function-calling API, which can return multiple functions each turn:

```
import datetime
from typing import Literal, Optional, Tuple

from langchain_core.pydantic_v1 import BaseModel, Field


class SubQuery(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    sub_query: str = Field(
        ...,
        description="A very specific query against the database.",
    )

```


```
from langchain.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \

Perform query decomposition. Given a user question, break it down into distinct sub questions that \
you need to answer in order to answer the original question.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
llm_with_tools = llm.bind_tools([SubQuery])
parser = PydanticToolsParser(tools=[SubQuery])
query_analyzer = prompt | llm_with_tools | parser

```


Let’s try it out:

```
query_analyzer.invoke({"question": "how to do rag"})

```


```
[SubQuery(sub_query='How to do rag')]

```


```
query_analyzer.invoke(
    {
        "question": "how to use multi-modal models in a chain and turn chain into a rest api"
    }
)

```


```
[SubQuery(sub_query='How to use multi-modal models in a chain?'),
 SubQuery(sub_query='How to turn a chain into a REST API?')]

```


```
query_analyzer.invoke(
    {
        "question": "what's the difference between web voyager and reflection agents? do they use langgraph?"
    }
)

```


```
[SubQuery(sub_query='What is Web Voyager and how does it differ from Reflection Agents?'),
 SubQuery(sub_query='Do Web Voyager and Reflection Agents use Langgraph?')]

```


Adding examples and tuning the prompt[​](#adding-examples-and-tuning-the-prompt "Direct link to Adding examples and tuning the prompt")
---------------------------------------------------------------------------------------------------------------------------------------

This works pretty well, but we probably want it to decompose the last question even further to separate the queries about Web Voyager and Reflection Agents. If we aren’t sure up front what types of queries will do best with our index, we can also intentionally include some redundancy in our queries, so that we return both sub queries and higher level queries.

To tune our query generation results, we can add some examples of inputs questions and gold standard output queries to our prompt. We can also try to improve our system message.

```
question = "What's chat langchain, is it a langchain template?"
queries = [
    SubQuery(sub_query="What is chat langchain"),
    SubQuery(sub_query="What is a langchain template"),
]
examples.append({"input": question, "tool_calls": queries})

```


```
question = "How would I use LangGraph to build an automaton"
queries = [
    SubQuery(sub_query="How to build automaton with LangGraph"),
]
examples.append({"input": question, "tool_calls": queries})

```


```
question = "How to build multi-agent system and stream intermediate steps from it"
queries = [
    SubQuery(sub_query="How to build multi-agent system"),
    SubQuery(sub_query="How to stream intermediate steps"),
    SubQuery(sub_query="How to stream intermediate steps from multi-agent system"),
]
examples.append({"input": question, "tool_calls": queries})

```


```
question = "What's the difference between LangChain agents and LangGraph?"
queries = [
    SubQuery(sub_query="What's the difference between LangChain agents and LangGraph?"),
    SubQuery(sub_query="What are LangChain agents"),
    SubQuery(sub_query="What is LangGraph"),
]
examples.append({"input": question, "tool_calls": queries})

```


Now we need to update our prompt template and chain so that the examples are included in each prompt. Since we’re working with OpenAI function-calling, we’ll need to do a bit of extra structuring to send example inputs and outputs to the model. We’ll create a `tool_example_to_messages` helper function to handle this for us:

```
import uuid
from typing import Dict, List

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def tool_example_to_messages(example: Dict) -> List[BaseMessage]:
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    openai_tool_calls = []
    for tool_call in example["tool_calls"]:
        openai_tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": tool_call.__class__.__name__,
                    "arguments": tool_call.json(),
                },
            }
        )
    messages.append(
        AIMessage(content="", additional_kwargs={"tool_calls": openai_tool_calls})
    )
    tool_outputs = example.get("tool_outputs") or [
        "This is an example of a correct usage of this tool. Make sure to continue using the tool this way."
    ] * len(openai_tool_calls)
    for output, tool_call in zip(tool_outputs, openai_tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return messages


example_msgs = [msg for ex in examples for msg in tool_example_to_messages(ex)]

```


```
from langchain_core.prompts import MessagesPlaceholder

system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \

Perform query decomposition. Given a user question, break it down into the most specific sub questions you can \
which will help you answer the original question. Each sub question should be about a single concept/fact/idea.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder("examples", optional=True),
        ("human", "{question}"),
    ]
)
query_analyzer_with_examples = (
    prompt.partial(examples=example_msgs) | llm_with_tools | parser
)

```


```
query_analyzer_with_examples.invoke(
    {
        "question": "what's the difference between web voyager and reflection agents? do they use langgraph?"
    }
)

```


```
[SubQuery(sub_query="What's the difference between web voyager and reflection agents"),
 SubQuery(sub_query='Do web voyager and reflection agents use LangGraph'),
 SubQuery(sub_query='What is web voyager'),
 SubQuery(sub_query='What are reflection agents')]

```
