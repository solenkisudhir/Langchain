# Routing | 🦜️🔗 LangChain
Sometimes we have multiple indexes for different domains, and for different questions we want to query different subsets of these indexes. For example, suppose we had one vector store index for all of the LangChain python documentation and one for all of the LangChain js documentation. Given a question about LangChain usage, we’d want to infer which language the the question was referring to and query the appropriate docs. **Query routing** is the process of classifying which index or subset of indexes a query should be performed on.

Setup[​](#setup "Direct link to Setup")
---------------------------------------

#### Install dependencies[​](#install-dependencies "Direct link to Install dependencies")

```
%pip install -qU langchain-core langchain-openai

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


Routing with function calling models[​](#routing-with-function-calling-models "Direct link to Routing with function calling models")
------------------------------------------------------------------------------------------------------------------------------------

With function-calling models it’s simple to use models for classification, which is what routing comes down to:

```
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )


llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to the appropriate data source.

Based on the programming language the question is referring to, route it to the relevant data source."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

router = prompt | structured_llm

```


```
/Users/bagatur/langchain/libs/core/langchain_core/_api/beta_decorator.py:86: LangChainBetaWarning: The function `with_structured_output` is in beta. It is actively being worked on, so the API may change.
  warn_beta(

```


```
question = """Why doesn't the following code work:

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""
router.invoke({"question": question})

```


```
RouteQuery(datasource='python_docs')

```


```
question = """Why doesn't the following code work:


import { ChatPromptTemplate } from "@langchain/core/prompts";


const chatPrompt = ChatPromptTemplate.fromMessages([
  ["human", "speak in {language}"],
]);

const formattedChatPrompt = await chatPrompt.invoke({
  input_language: "french"
});
"""
router.invoke({"question": question})

```


```
RouteQuery(datasource='js_docs')

```


Routing to multiple indexes[​](#routing-to-multiple-indexes "Direct link to Routing to multiple indexes")
---------------------------------------------------------------------------------------------------------

If we may want to query multiple indexes we can do that, too, by updating our schema to accept a List of data sources:

```
from typing import List


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasources: List[Literal["python_docs", "js_docs", "golang_docs"]] = Field(
        ...,
        description="Given a user question choose which datasources would be most relevant for answering their question",
    )


llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery)
router = prompt | structured_llm
router.invoke(
    {
        "question": "is there feature parity between the Python and JS implementations of OpenAI chat models"
    }
)

```


```
RouteQuery(datasources=['python_docs', 'js_docs'])

```
