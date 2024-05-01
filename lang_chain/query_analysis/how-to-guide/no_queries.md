# Handle Cases Where No Queries are Generated | 🦜️🔗 LangChain
Sometimes, a query analysis technique may allow for any number of queries to be generated - including no queries! In this case, our overall chain will need to inspect the result of the query analysis before deciding whether to call the retriever or not.

We will use mock data for this example.

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
vectorstore = Chroma.from_texts(
    texts,
    embeddings,
)
retriever = vectorstore.as_retriever()

```


Query analysis[​](#query-analysis "Direct link to Query analysis")
------------------------------------------------------------------

We will use function calling to structure the output. However, we will configure the LLM such that is doesn’t NEED to call the function representing a search query (should it decide not to). We will also then use a prompt to do query analysis that explicitly lays when it should and shouldn’t make a search.

```
from typing import Optional

from langchain_core.pydantic_v1 import BaseModel, Field


class Search(BaseModel):
    """Search over a database of job records."""

    query: str = Field(
        ...,
        description="Similarity search query applied to job record.",
    )

```


```
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

system = """You have the ability to issue search queries to get information to help answer user information.

You do not NEED to look things up. If you don't need to, then just respond normally."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.bind_tools([Search])
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm

```


We can see that by invoking this we get an message that sometimes - but not always - returns a tool call.

```
query_analyzer.invoke("where did Harrison Work")

```


```
AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_ZnoVX4j9Mn8wgChaORyd1cvq', 'function': {'arguments': '{"query":"Harrison"}', 'name': 'Search'}, 'type': 'function'}]})

```


```
query_analyzer.invoke("hi!")

```


```
AIMessage(content='Hello! How can I assist you today?')

```


Retrieval with query analysis[​](#retrieval-with-query-analysis "Direct link to Retrieval with query analysis")
---------------------------------------------------------------------------------------------------------------

So how would we include this in a chain? Let’s look at an example below.

```
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.runnables import chain

output_parser = PydanticToolsParser(tools=[Search])

```


```
@chain
def custom_chain(question):
    response = query_analyzer.invoke(question)
    if "tool_calls" in response.additional_kwargs:
        query = output_parser.invoke(response)
        docs = retriever.invoke(query[0].query)
        # Could add more logic - like another LLM call - here
        return docs
    else:
        return response

```


```
custom_chain.invoke("where did Harrison Work")

```


```
Number of requested results 4 is greater than number of elements in index 1, updating n_results = 1

```


```
[Document(page_content='Harrison worked at Kensho')]

```


```
custom_chain.invoke("hi!")

```


```
AIMessage(content='Hello! How can I assist you today?')

```


* * *

#### Help us out by providing feedback on this documentation page: