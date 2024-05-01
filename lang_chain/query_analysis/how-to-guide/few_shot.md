# Add Examples to the Prompt | 🦜️🔗 LangChain
As our query analysis becomes more complex, the LLM may struggle to understand how exactly it should respond in certain scenarios. In order to improve performance here, we can add examples to the prompt to guide the LLM.

Let’s take a look at how we can add examples for the LangChain YouTube video query analyzer we built in the [Quickstart](https://python.langchain.com/docs/use_cases/query_analysis/quickstart/).

Setup[​](#setup "Direct link to Setup")
---------------------------------------

#### Install dependencies[​](#install-dependencies "Direct link to Install dependencies")

```
# %pip install -qU langchain-core langchain-openai

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


Query schema[​](#query-schema "Direct link to Query schema")
------------------------------------------------------------

We’ll define a query schema that we want our model to output. To make our query analysis a bit more interesting, we’ll add a `sub_queries` field that contains more narrow questions derived from the top level question.

```
from typing import List, Optional

from langchain_core.pydantic_v1 import BaseModel, Field

sub_queries_description = """\
If the original question contains multiple distinct sub-questions, \
or if there are more generic questions that would be helpful to answer in \
order to answer the original question, write a list of all relevant sub-questions. \
Make sure this list is comprehensive and covers all parts of the original question. \
It's ok if there's redundancy in the sub-questions. \
Make sure the sub-questions are as narrowly focused as possible."""


class Search(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    query: str = Field(
        ...,
        description="Primary similarity search query applied to video transcripts.",
    )
    sub_queries: List[str] = Field(
        default_factory=list, description=sub_queries_description
    )
    publish_year: Optional[int] = Field(None, description="Year video was published")

```


Query generation[​](#query-generation "Direct link to Query generation")
------------------------------------------------------------------------

```
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a list of database queries optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder("examples", optional=True),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(Search)
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm

```


Let’s try out our query analyzer without any examples in the prompt:

```
query_analyzer.invoke(
    "what's the difference between web voyager and reflection agents? do both use langgraph?"
)

```


```
Search(query='web voyager vs reflection agents', sub_queries=['difference between web voyager and reflection agents', 'do web voyager and reflection agents use langgraph'], publish_year=None)

```


Adding examples and tuning the prompt[​](#adding-examples-and-tuning-the-prompt "Direct link to Adding examples and tuning the prompt")
---------------------------------------------------------------------------------------------------------------------------------------

This works pretty well, but we probably want it to decompose the question even further to separate the queries about Web Voyager and Reflection Agents.

To tune our query generation results, we can add some examples of inputs questions and gold standard output queries to our prompt.

```
question = "What's chat langchain, is it a langchain template?"
query = Search(
    query="What is chat langchain and is it a langchain template?",
    sub_queries=["What is chat langchain", "What is a langchain template"],
)
examples.append({"input": question, "tool_calls": [query]})

```


```
question = "How to build multi-agent system and stream intermediate steps from it"
query = Search(
    query="How to build multi-agent system and stream intermediate steps from it",
    sub_queries=[
        "How to build multi-agent system",
        "How to stream intermediate steps from multi-agent system",
        "How to stream intermediate steps",
    ],
)

examples.append({"input": question, "tool_calls": [query]})

```


```
question = "LangChain agents vs LangGraph?"
query = Search(
    query="What's the difference between LangChain agents and LangGraph? How do you deploy them?",
    sub_queries=[
        "What are LangChain agents",
        "What is LangGraph",
        "How do you deploy LangChain agents",
        "How do you deploy LangGraph",
    ],
)
examples.append({"input": question, "tool_calls": [query]})

```


Now we need to update our prompt template and chain so that the examples are included in each prompt. Since we’re working with OpenAI function-calling, we’ll need to do a bit of extra structuring to send example inputs and outputs to the model. We’ll create a `tool_example_to_messages` helper function to handle this for us:

```
import uuid
from typing import Dict

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
        "You have correctly called this tool."
    ] * len(openai_tool_calls)
    for output, tool_call in zip(tool_outputs, openai_tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return messages


example_msgs = [msg for ex in examples for msg in tool_example_to_messages(ex)]

```


```
from langchain_core.prompts import MessagesPlaceholder

query_analyzer_with_examples = (
    {"question": RunnablePassthrough()}
    | prompt.partial(examples=example_msgs)
    | structured_llm
)

```


```
query_analyzer_with_examples.invoke(
    "what's the difference between web voyager and reflection agents? do both use langgraph?"
)

```


```
Search(query='Difference between web voyager and reflection agents, do they both use LangGraph?', sub_queries=['What is Web Voyager', 'What are Reflection agents', 'Do Web Voyager and Reflection agents use LangGraph'], publish_year=None)

```


Thanks to our examples we get a slightly more decomposed search query. With some more prompt engineering and tuning of our examples we could improve query generation even more.

You can see that the examples are passed to the model as messages in the [LangSmith trace](https://smith.langchain.com/public/aeaaafce-d2b1-4943-9a61-bc954e8fc6f2/r).