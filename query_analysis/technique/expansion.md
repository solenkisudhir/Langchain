# Expansion | 🦜️🔗 LangChain
Information retrieval systems can be sensitive to phrasing and specific keywords. To mitigate this, one classic retrieval technique is to generate multiple paraphrased versions of a query and return results for all versions of the query. This is called **query expansion**. LLMs are a great tool for generating these alternate versions of a query.

Let’s take a look at how we might do query expansion for our Q&A bot over the LangChain YouTube videos, which we started in the [Quickstart](https://python.langchain.com/docs/use_cases/query_analysis/quickstart/).

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

To make sure we get multiple paraphrasings we’ll use OpenAI’s function-calling API.

```
from langchain_core.pydantic_v1 import BaseModel, Field


class ParaphrasedQuery(BaseModel):
    """You have performed query expansion to generate a paraphrasing of a question."""

    paraphrased_query: str = Field(
        ...,
        description="A unique paraphrasing of the original question.",
    )

```


```
from langchain.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \

Perform query expansion. If there are multiple common ways of phrasing a user question \
or common synonyms for key words in the question, make sure to return multiple versions \
of the query with the different phrasings.

If there are acronyms or words you are not familiar with, do not try to rephrase them.

Return at least 3 versions of the question."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
llm_with_tools = llm.bind_tools([ParaphrasedQuery])
query_analyzer = prompt | llm_with_tools | PydanticToolsParser(tools=[ParaphrasedQuery])

```


Let’s see what queries our analyzer generates for the questions we searched earlier:

```
query_analyzer.invoke(
    {
        "question": "how to use multi-modal models in a chain and turn chain into a rest api"
    }
)

```


```
[ParaphrasedQuery(paraphrased_query='How to utilize multi-modal models sequentially and convert the sequence into a REST API'),
 ParaphrasedQuery(paraphrased_query='Steps for using multi-modal models in a series and transforming the series into a RESTful API'),
 ParaphrasedQuery(paraphrased_query='Guide on employing multi-modal models in a chain and converting the chain into a RESTful API')]

```


```
query_analyzer.invoke({"question": "stream events from llm agent"})

```


```
[ParaphrasedQuery(paraphrased_query='How to stream events from LLM agent?'),
 ParaphrasedQuery(paraphrased_query='How can I receive events from LLM agent in real-time?'),
 ParaphrasedQuery(paraphrased_query='What is the process for capturing events from LLM agent?')]

```
