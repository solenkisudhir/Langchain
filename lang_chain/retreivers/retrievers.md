# Retrievers | 🦜️🔗 LangChain
A retriever is an interface that returns documents given an unstructured query. It is more general than a vector store. A retriever does not need to be able to store documents, only to return (or retrieve) them. Vector stores can be used as the backbone of a retriever, but there are other types of retrievers as well.

Retrievers accept a string query as input and return a list of `Document`'s as output.

Advanced Retrieval Types[​](#advanced-retrieval-types "Direct link to Advanced Retrieval Types")
------------------------------------------------------------------------------------------------

LangChain provides several advanced retrieval types. A full list is below, along with the following information:

**Name**: Name of the retrieval algorithm.

**Index Type**: Which index type (if any) this relies on.

**Uses an LLM**: Whether this retrieval method uses an LLM.

**When to Use**: Our commentary on when you should considering using this retrieval method.

**Description**: Description of what this retrieval algorithm is doing.



* Name: Vectorstore
  * Index Type: Vectorstore
  * Uses an LLM: No
  * When to Use: If you are just getting started and looking for something quick and easy.
  * Description: This is the simplest method and the one that is easiest to get started with. It involves creating embeddings for each piece of text.
* Name: ParentDocument
  * Index Type: Vectorstore + Document Store
  * Uses an LLM: No
  * When to Use: If your pages have lots of smaller pieces of distinct information that are best indexed by themselves, but best retrieved all together.
  * Description: This involves indexing multiple chunks for each document. Then you find the chunks that are most similar in embedding space, but you retrieve the whole parent document and return that (rather than individual chunks).
* Name: Multi Vector
  * Index Type: Vectorstore + Document Store
  * Uses an LLM: Sometimes during indexing
  * When to Use: If you are able to extract information from documents that you think is more relevant to index than the text itself.
  * Description: This involves creating multiple vectors for each document. Each vector could be created in a myriad of ways - examples include summaries of the text and hypothetical questions.
* Name: Self Query
  * Index Type: Vectorstore
  * Uses an LLM: Yes
  * When to Use: If users are asking questions that are better answered by fetching documents based on metadata rather than similarity with the text.
  * Description: This uses an LLM to transform user input into two things: (1) a string to look up semantically, (2) a metadata filer to go along with it. This is useful because oftentimes questions are about the METADATA of documents (not the content itself).
* Name: Contextual Compression
  * Index Type: Any
  * Uses an LLM: Sometimes
  * When to Use: If you are finding that your retrieved documents contain too much irrelevant information and are distracting the LLM.
  * Description: This puts a post-processing step on top of another retriever and extracts only the most relevant information from retrieved documents. This can be done with embeddings or an LLM.
* Name: Time-Weighted Vectorstore
  * Index Type: Vectorstore
  * Uses an LLM: No
  * When to Use: If you have timestamps associated with your documents, and you want to retrieve the most recent ones
  * Description: This fetches documents based on a combination of semantic similarity (as in normal vector retrieval) and recency (looking at timestamps of indexed documents)
* Name: Multi-Query Retriever
  * Index Type: Any
  * Uses an LLM: Yes
  * When to Use: If users are asking questions that are complex and require multiple pieces of distinct information to respond
  * Description: This uses an LLM to generate multiple queries from the original one. This is useful when the original query needs pieces of information about multiple topics to be properly answered. By generating multiple queries, we can then fetch documents for each of them.
* Name: Ensemble
  * Index Type: Any
  * Uses an LLM: No
  * When to Use: If you have multiple retrieval methods and want to try combining them.
  * Description: This fetches documents from multiple retrievers and then combines them.
* Name: Long-Context Reorder
  * Index Type: Any
  * Uses an LLM: No
  * When to Use: If you are working with a long-context model and noticing that it's not paying attention to information in the middle of retrieved documents.
  * Description: This fetches documents from an underlying retriever, and then reorders them so that the most similar are near the beginning and end. This is useful because it's been shown that for longer context models they sometimes don't pay attention to information in the middle of the context window.


[Third Party Integrations](https://python.langchain.com/docs/integrations/retrievers/)[​](#third-party-integrations "Direct link to third-party-integrations")
--------------------------------------------------------------------------------------------------------------------------------------------------------------

LangChain also integrates with many third-party retrieval services. For a full list of these, check out [this list](https://python.langchain.com/docs/integrations/retrievers/) of all integrations.

Using Retrievers in LCEL[​](#using-retrievers-in-lcel "Direct link to Using Retrievers in LCEL")
------------------------------------------------------------------------------------------------

Since retrievers are `Runnable`'s, we can easily compose them with other `Runnable` objects:

```
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

template = """Answer the question based only on the following context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

chain.invoke("What did the president say about technology?")


```


Custom Retriever[​](#custom-retriever "Direct link to Custom Retriever")
------------------------------------------------------------------------

See the [documentation here](https://python.langchain.com/docs/modules/data_connection/retrievers/custom_retriever/) to implement a custom retriever.