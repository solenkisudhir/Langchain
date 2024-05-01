# Structuring | 🦜️🔗 LangChain
One of the most important steps in retrieval is turning a text input into the right search and filter parameters. This process of extracting structured parameters from an unstructured input is what we refer to as **query structuring**.

To illustrate, let’s return to our example of a Q&A bot over the LangChain YouTube videos from the [Quickstart](https://python.langchain.com/docs/use_cases/query_analysis/quickstart/) and see what more complex structured queries might look like in this case.

Setup[​](#setup "Direct link to Setup")
---------------------------------------

#### Install dependencies[​](#install-dependencies "Direct link to Install dependencies")

```
# %pip install -qU langchain langchain-openai youtube-transcript-api pytube

```


#### Set environment variables[​](#set-environment-variables "Direct link to Set environment variables")

We’ll use OpenAI in this example:

```
import getpass
import os

# os.environ["OPENAI_API_KEY"] = getpass.getpass()

# Optional, uncomment to trace runs with LangSmith. Sign up here: https://smith.langchain.com.
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

```


### Load example document[​](#load-example-document "Direct link to Load example document")

Let’s load a representative document

```
from langchain_community.document_loaders import YoutubeLoader

docs = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=pbAd8O1Lvm4", add_video_info=True
).load()

```


Here’s the metadata associated with a video:

```
{'source': 'pbAd8O1Lvm4',
 'title': 'Self-reflective RAG with LangGraph: Self-RAG and CRAG',
 'description': 'Unknown',
 'view_count': 9006,
 'thumbnail_url': 'https://i.ytimg.com/vi/pbAd8O1Lvm4/hq720.jpg',
 'publish_date': '2024-02-07 00:00:00',
 'length': 1058,
 'author': 'LangChain'}

```


And here’s a sample from a document’s contents:

```
docs[0].page_content[:500]

```


```
"hi this is Lance from Lang chain I'm going to be talking about using Lang graph to build a diverse and sophisticated rag flows so just to set the stage the basic rag flow you can see here starts with a question retrieval of relevant documents from an index which are passed into the context window of an llm for generation of an answer grounded in the ret documents so that's kind of the basic outline and we can see it's like a very linear path um in practice though you often encounter a few differ"

```


Query schema[​](#query-schema "Direct link to Query schema")
------------------------------------------------------------

In order to generate structured queries we first need to define our query schema. We can see that each document has a title, view count, publication date, and length in seconds. Let’s assume we’ve built an index that allows us to perform unstructured search over the contents and title of each document, and to use range filtering on view count, publication date, and length.

To start we’ll create a schema with explicit min and max attributes for view count, publication date, and video length so that those can be filtered on. And we’ll add separate attributes for searches against the transcript contents versus the video title.

We could alternatively create a more generic schema where instead of having one or more filter attributes for each filterable field, we have a single `filters` attribute that takes a list of (attribute, condition, value) tuples. We’ll demonstrate how to do this as well. Which approach works best depends on the complexity of your index. If you have many filterable fields then it may be better to have a single `filters` query attribute. If you have only a few filterable fields and/or there are fields that can only be filtered in very specific ways, it can be helpful to have separate query attributes for them, each with their own description.

```
import datetime
from typing import Literal, Optional, Tuple

from langchain_core.pydantic_v1 import BaseModel, Field


class TutorialSearch(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    content_search: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )
    title_search: str = Field(
        ...,
        description=(
            "Alternate version of the content search query to apply to video titles. "
            "Should be succinct and only include key words that could be in a video "
            "title."
        ),
    )
    min_view_count: Optional[int] = Field(
        None,
        description="Minimum view count filter, inclusive. Only use if explicitly specified.",
    )
    max_view_count: Optional[int] = Field(
        None,
        description="Maximum view count filter, exclusive. Only use if explicitly specified.",
    )
    earliest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Earliest publish date filter, inclusive. Only use if explicitly specified.",
    )
    latest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Latest publish date filter, exclusive. Only use if explicitly specified.",
    )
    min_length_sec: Optional[int] = Field(
        None,
        description="Minimum video length in seconds, inclusive. Only use if explicitly specified.",
    )
    max_length_sec: Optional[int] = Field(
        None,
        description="Maximum video length in seconds, exclusive. Only use if explicitly specified.",
    )

    def pretty_print(self) -> None:
        for field in self.__fields__:
            if getattr(self, field) is not None and getattr(self, field) != getattr(
                self.__fields__[field], "default", None
            ):
                print(f"{field}: {getattr(self, field)}")

```


Query generation[​](#query-generation "Direct link to Query generation")
------------------------------------------------------------------------

To convert user questions to structured queries we’ll make use of a function-calling model, like ChatOpenAI. LangChain has some nice constructors that make it easy to specify a desired function call schema via a Pydantic class:

```
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a database query optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(TutorialSearch)
query_analyzer = prompt | structured_llm

```


Let’s try it out:

```
query_analyzer.invoke({"question": "rag from scratch"}).pretty_print()

```


```
content_search: rag from scratch
title_search: rag from scratch

```


```
query_analyzer.invoke(
    {"question": "videos on chat langchain published in 2023"}
).pretty_print()

```


```
content_search: chat langchain
title_search: chat langchain
earliest_publish_date: 2023-01-01
latest_publish_date: 2024-01-01

```


```
query_analyzer.invoke(
    {
        "question": "how to use multi-modal models in an agent, only videos under 5 minutes"
    }
).pretty_print()

```


```
content_search: multi-modal models agent
title_search: multi-modal models agent
max_length_sec: 300

```


Alternative: Succinct schema[​](#alternative-succinct-schema "Direct link to Alternative: Succinct schema")
-----------------------------------------------------------------------------------------------------------

If we have many filterable fields then having a verbose schema could harm performance, or may not even be possible given limitations on the size of function schemas. In these cases we can try more succinct query schemas that trade off some explicitness of direction for concision:

```
from typing import List, Literal, Union


class Filter(BaseModel):
    field: Literal["view_count", "publish_date", "length_sec"]
    comparison: Literal["eq", "lt", "lte", "gt", "gte"]
    value: Union[int, datetime.date] = Field(
        ...,
        description="If field is publish_date then value must be a ISO-8601 format date",
    )


class TutorialSearch(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    content_search: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )
    title_search: str = Field(
        ...,
        description=(
            "Alternate version of the content search query to apply to video titles. "
            "Should be succinct and only include key words that could be in a video "
            "title."
        ),
    )
    filters: List[Filter] = Field(
        default_factory=list,
        description="Filters over specific fields. Final condition is a logical conjunction of all filters.",
    )

    def pretty_print(self) -> None:
        for field in self.__fields__:
            if getattr(self, field) is not None and getattr(self, field) != getattr(
                self.__fields__[field], "default", None
            ):
                print(f"{field}: {getattr(self, field)}")

```


```
structured_llm = llm.with_structured_output(TutorialSearch)
query_analyzer = prompt | structured_llm

```


Let’s try it out:

```
query_analyzer.invoke({"question": "rag from scratch"}).pretty_print()

```


```
content_search: rag from scratch
title_search: rag
filters: []

```


```
query_analyzer.invoke(
    {"question": "videos on chat langchain published in 2023"}
).pretty_print()

```


```
content_search: chat langchain
title_search: 2023
filters: [Filter(field='publish_date', comparison='eq', value=datetime.date(2023, 1, 1))]

```


```
query_analyzer.invoke(
    {
        "question": "how to use multi-modal models in an agent, only videos under 5 minutes and with over 276 views"
    }
).pretty_print()

```


```
content_search: multi-modal models in an agent
title_search: multi-modal models agent
filters: [Filter(field='length_sec', comparison='lt', value=300), Filter(field='view_count', comparison='gte', value=276)]

```


We can see that the analyzer handles integers well but struggles with date ranges. We can try adjusting our schema description and/or our prompt to correct this:

```
class TutorialSearch(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    content_search: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )
    title_search: str = Field(
        ...,
        description=(
            "Alternate version of the content search query to apply to video titles. "
            "Should be succinct and only include key words that could be in a video "
            "title."
        ),
    )
    filters: List[Filter] = Field(
        default_factory=list,
        description=(
            "Filters over specific fields. Final condition is a logical conjunction of all filters. "
            "If a time period longer than one day is specified then it must result in filters that define a date range. "
            f"Keep in mind the current date is {datetime.date.today().strftime('%m-%d-%Y')}."
        ),
    )

    def pretty_print(self) -> None:
        for field in self.__fields__:
            if getattr(self, field) is not None and getattr(self, field) != getattr(
                self.__fields__[field], "default", None
            ):
                print(f"{field}: {getattr(self, field)}")


structured_llm = llm.with_structured_output(TutorialSearch)
query_analyzer = prompt | structured_llm

```


```
query_analyzer.invoke(
    {"question": "videos on chat langchain published in 2023"}
).pretty_print()

```


```
content_search: chat langchain
title_search: chat langchain
filters: [Filter(field='publish_date', comparison='gte', value=datetime.date(2023, 1, 1)), Filter(field='publish_date', comparison='lte', value=datetime.date(2023, 12, 31))]

```


This seems to work!

Sorting: Going beyond search[​](#sorting-going-beyond-search "Direct link to Sorting: Going beyond search")
-----------------------------------------------------------------------------------------------------------

With certain indexes searching by field isn’t the only way to retrieve results — we can also sort documents by a field and retrieve the top sorted results. With structured querying this is easy to accomodate by adding separate query fields that specify how to sort results.

```
class TutorialSearch(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    content_search: str = Field(
        "",
        description="Similarity search query applied to video transcripts.",
    )
    title_search: str = Field(
        "",
        description=(
            "Alternate version of the content search query to apply to video titles. "
            "Should be succinct and only include key words that could be in a video "
            "title."
        ),
    )
    min_view_count: Optional[int] = Field(
        None, description="Minimum view count filter, inclusive."
    )
    max_view_count: Optional[int] = Field(
        None, description="Maximum view count filter, exclusive."
    )
    earliest_publish_date: Optional[datetime.date] = Field(
        None, description="Earliest publish date filter, inclusive."
    )
    latest_publish_date: Optional[datetime.date] = Field(
        None, description="Latest publish date filter, exclusive."
    )
    min_length_sec: Optional[int] = Field(
        None, description="Minimum video length in seconds, inclusive."
    )
    max_length_sec: Optional[int] = Field(
        None, description="Maximum video length in seconds, exclusive."
    )
    sort_by: Literal[
        "relevance",
        "view_count",
        "publish_date",
        "length",
    ] = Field("relevance", description="Attribute to sort by.")
    sort_order: Literal["ascending", "descending"] = Field(
        "descending", description="Whether to sort in ascending or descending order."
    )

    def pretty_print(self) -> None:
        for field in self.__fields__:
            if getattr(self, field) is not None and getattr(self, field) != getattr(
                self.__fields__[field], "default", None
            ):
                print(f"{field}: {getattr(self, field)}")


structured_llm = llm.with_structured_output(TutorialSearch)
query_analyzer = prompt | structured_llm

```


```
query_analyzer.invoke(
    {"question": "What has LangChain released lately?"}
).pretty_print()

```


```
title_search: LangChain
sort_by: publish_date

```


```
query_analyzer.invoke({"question": "What are the longest videos?"}).pretty_print()

```


We can even support searching and sorting together. This might look like first retrieving all results above a relevancy threshold and then sorting them according to the specified attribute:

```
query_analyzer.invoke(
    {"question": "What are the shortest videos about agents?"}
).pretty_print()

```


```
content_search: agents
sort_by: length
sort_order: ascending

```
