# Handle Multiple Queries | 🦜️🔗 LangChain
Sometimes, a query analysis technique may allow for multiple queries to be generated. In these cases, we need to remember to run all queries and then to combine the results. We will show a simple example (using mock data) of how to do that.

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

texts = ["Harrison worked at Kensho", "Ankush worked at Facebook"]
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_texts(
    texts,
    embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

```


Query analysis[​](#query-analysis "Direct link to Query analysis")
------------------------------------------------------------------

We will use function calling to structure the output. We will let it return multiple queries.

```
from typing import List, Optional

from langchain_core.pydantic_v1 import BaseModel, Field


class Search(BaseModel):
    """Search over a database of job records."""

    queries: List[str] = Field(
        ...,
        description="Distinct queries to search for",
    )

```


```
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

output_parser = PydanticToolsParser(tools=[Search])

system = """You have the ability to issue search queries to get information to help answer user information.

If you need to look up two distinct pieces of information, you are allowed to do that!"""
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


```
/Users/harrisonchase/workplace/langchain/libs/core/langchain_core/_api/beta_decorator.py:86: LangChainBetaWarning: The function `with_structured_output` is in beta. It is actively being worked on, so the API may change.
  warn_beta(

```


We can see that this allows for creating multiple queries

```
query_analyzer.invoke("where did Harrison Work")

```


```
Search(queries=['Harrison work location'])

```


```
query_analyzer.invoke("where did Harrison and ankush Work")

```


```
Search(queries=['Harrison work place', 'Ankush work place'])

```


Retrieval with query analysis[​](#retrieval-with-query-analysis "Direct link to Retrieval with query analysis")
---------------------------------------------------------------------------------------------------------------

So how would we include this in a chain? One thing that will make this a lot easier is if we call our retriever asyncronously - this will let us loop over the queries and not get blocked on the response time.

```
from langchain_core.runnables import chain

```


```
@chain
async def custom_chain(question):
    response = await query_analyzer.ainvoke(question)
    docs = []
    for query in response.queries:
        new_docs = await retriever.ainvoke(query)
        docs.extend(new_docs)
    # You probably want to think about reranking or deduplicating documents here
    # But that is a separate topic
    return docs

```


```
await custom_chain.ainvoke("where did Harrison Work")

```


```
[Document(page_content='Harrison worked at Kensho')]

```


```
await custom_chain.ainvoke("where did Harrison and ankush Work")

```


```
[Document(page_content='Harrison worked at Kensho'),
 Document(page_content='Ankush worked at Facebook')]

```
