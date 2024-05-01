# Quickstart | 🦜️🔗 LangChain
[🦜️🔗](#)

*   [LangSmith](https://smith.langchain.com/)
*   [LangSmith Docs](https://docs.smith.langchain.com/)
*   [LangServe GitHub](https://github.com/langchain-ai/langserve)
*   [Templates GitHub](https://github.com/langchain-ai/langchain/tree/master/templates)
*   [Templates Hub](https://templates.langchain.com/)
*   [LangChain Hub](https://smith.langchain.com/hub)
*   [JS/TS Docs](https://js.langchain.com/)

[💬](https://chat.langchain.com/)[](https://github.com/langchain-ai/langchain)

*   [](https://python.langchain.com/)
*   [Use cases](https://python.langchain.com/docs/use_cases/)
*   [Extracting structured output](https://python.langchain.com/docs/use_cases/extraction/)
*   Quickstart

Quickstart
----------

In this quick start, we will use [chat models](https://python.langchain.com/docs/modules/model_io/chat/) that are capable of **function/tool calling** to extract information from text.

Set up[​](#set-up "Direct link to Set up")
------------------------------------------

We will use the [structured output](https://python.langchain.com/docs/modules/model_io/chat/structured_output/) method available on LLMs that are capable of **function/tool calling**.

Select a model, install the dependencies for it and set up API keys!

```
!pip install langchain

# Install a model capable of tool calling
# pip install langchain-openai
# pip install langchain-mistralai
# pip install langchain-fireworks

# Set env vars for the relevant model or load from a .env file:
# import dotenv
# dotenv.load_dotenv()

```


The Schema[​](#the-schema "Direct link to The Schema")
------------------------------------------------------

First, we need to describe what information we want to extract from the text.

We’ll use Pydantic to define an example schema to extract personal information.

```
from typing import Optional

from langchain_core.pydantic_v1 import BaseModel, Field


class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the peron's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )

```


There are two best practices when defining schema:

1.  Document the **attributes** and the **schema** itself: This information is sent to the LLM and is used to improve the quality of information extraction.
2.  Do not force the LLM to make up information! Above we used `Optional` for the attributes allowing the LLM to output `None` if it doesn’t know the answer.

For best performance, document the schema well and make sure the model isn’t force to return results if there’s no information to be extracted in the text.

The Extractor[​](#the-extractor "Direct link to The Extractor")
---------------------------------------------------------------

Let’s create an information extractor using the schema we defined above.

```
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)

```


We need to use a model that supports function/tool calling.

Please review [structured output](https://python.langchain.com/docs/modules/model_io/chat/structured_output/) for list of some models that can be used with this API.

```
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(model="mistral-large-latest", temperature=0)

runnable = prompt | llm.with_structured_output(schema=Person)

```


Let’s test it out

```
text = "Alan Smith is 6 feet tall and has blond hair."
runnable.invoke({"text": text})

```


```
Person(name='Alan Smith', hair_color='blond', height_in_meters='1.8288')

```


Extraction is Generative 🤯

LLMs are generative models, so they can do some pretty cool things like correctly extract the height of the person in meters even though it was provided in feet!

Multiple Entities[​](#multiple-entities "Direct link to Multiple Entities")
---------------------------------------------------------------------------

In **most cases**, you should be extracting a list of entities rather than a single entity.

This can be easily achieved using pydantic by nesting models inside one another.

```
from typing import List, Optional

from langchain_core.pydantic_v1 import BaseModel, Field


class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the peron's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )


class Data(BaseModel):
    """Extracted data about people."""

    # Creates a model so that we can extract multiple entities.
    people: List[Person]

```


Extraction might not be perfect here. Please continue to see how to use **Reference Examples** to improve the quality of extraction, and see the **guidelines** section!

```
runnable = prompt | llm.with_structured_output(schema=Data)
text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
runnable.invoke({"text": text})

```


```
Data(people=[Person(name='Jeff', hair_color=None, height_in_meters=None), Person(name='Anna', hair_color=None, height_in_meters=None)])

```


When the schema accommodates the extraction of **multiple entities**, it also allows the model to extract **no entities** if no relevant information is in the text by providing an empty list.

This is usually a **good** thing! It allows specifying **required** attributes on an entity without necessarily forcing the model to detect this entity.

Next steps[​](#next-steps "Direct link to Next steps")
------------------------------------------------------

Now that you understand the basics of extraction with LangChain, you’re ready to proceed to the rest of the how-to guide:

*   [Add Examples](https://python.langchain.com/docs/use_cases/extraction/how_to/examples/): Learn how to use **reference examples** to improve performance.
*   [Handle Long Text](https://python.langchain.com/docs/use_cases/extraction/how_to/handle_long_text/): What should you do if the text does not fit into the context window of the LLM?
*   [Handle Files](https://python.langchain.com/docs/use_cases/extraction/how_to/handle_files/): Examples of using LangChain document loaders and parsers to extract from files like PDFs.
*   [Use a Parsing Approach](https://python.langchain.com/docs/use_cases/extraction/how_to/parse/): Use a prompt based approach to extract with models that do not support **tool/function calling**.
*   [Guidelines](https://python.langchain.com/docs/use_cases/extraction/guidelines/): Guidelines for getting good performance on extraction tasks.

* * *

#### Help us out by providing feedback on this documentation page:

[

Extracting structured output

](https://python.langchain.com/docs/use_cases/extraction/)[](https://python.langchain.com/docs/use_cases/extraction/guidelines/)

*   [Set up](#set-up)
*   [The Schema](#the-schema)
*   [The Extractor](#the-extractor)
*   [Multiple Entities](#multiple-entities)
*   [Next steps](#next-steps)