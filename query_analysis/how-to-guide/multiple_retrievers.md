# Handle Multiple Retrievers | 🦜️🔗 LangChain
Sometimes, a query analysis technique may allow for selection of which retriever to use. To use this, you will need to add some logic to select the retriever to do. We will show a simple example (using mock data) of how to do that.

Setup[​](#setup "Direct link to Setup")
---------------------------------------

#### Install dependencies[​](#install-dependencies "Direct link to Install dependencies")

```
# %pip install -qU langchain langchain-community langchain-openai langchain-chroma

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


### Create Index[​](#create-index "Direct link to Create Index")

We will create a vectorstore over fake information.

```
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

texts = ["Harrison worked at Kensho"]
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_texts(texts, embeddings, collection_name="harrison")
retriever_harrison = vectorstore.as_retriever(search_kwargs={"k": 1})

texts = ["Ankush worked at Facebook"]
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_texts(texts, embeddings, collection_name="ankush")
retriever_ankush = vectorstore.as_retriever(search_kwargs={"k": 1})

```


Query analysis[​](#query-analysis "Direct link to Query analysis")
------------------------------------------------------------------

We will use function calling to structure the output. We will let it return multiple queries.

```
from typing import List, Optional

from langchain_core.pydantic_v1 import BaseModel, Field


class Search(BaseModel):
    """Search for information about a person."""

    query: str = Field(
        ...,
        description="Query to look up",
    )
    person: str = Field(
        ...,
        description="Person to look things up for. Should be `HARRISON` or `ANKUSH`.",
    )

```


```
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

output_parser = PydanticToolsParser(tools=[Search])

system = """You have the ability to issue search queries to get information to help answer user information."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(Search)
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm

```


We can see that this allows for routing between retrievers

```
query_analyzer.invoke("where did Harrison Work")

```


```
Search(query='workplace', person='HARRISON')

```


```
query_analyzer.invoke("where did ankush Work")

```


```
Search(query='workplace', person='ANKUSH')

```


Retrieval with query analysis[​](#retrieval-with-query-analysis "Direct link to Retrieval with query analysis")
---------------------------------------------------------------------------------------------------------------

So how would we include this in a chain? We just need some simple logic to select the retriever and pass in the search query

```
from langchain_core.runnables import chain

```


```
retrievers = {
    "HARRISON": retriever_harrison,
    "ANKUSH": retriever_ankush,
}

```


```
@chain
def custom_chain(question):
    response = query_analyzer.invoke(question)
    retriever = retrievers[response.person]
    return retriever.invoke(response.query)

```


```
custom_chain.invoke("where did Harrison Work")

```


```
[Document(page_content='Harrison worked at Kensho')]

```


```
custom_chain.invoke("where did ankush Work")

```


```
[Document(page_content='Ankush worked at Facebook')]

```
