# Parsing | 🦜️🔗 LangChain
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
*   More
*   Parsing

Parsing
-------

LLMs that are able to follow prompt instructions well can be tasked with outputting information in a given format.

This approach relies on designing good prompts and then parsing the output of the LLMs to make them extract information well.

Here, we’ll use Claude which is great at following instructions! See [Anthropic models](https://www.anthropic.com/api).

```
from langchain_anthropic.chat_models import ChatAnthropic

model = ChatAnthropic(model_name="claude-3-sonnet-20240229", temperature=0)

```


All the same considerations for extraction quality apply for parsing approach. Review the [guidelines](https://python.langchain.com/docs/use_cases/extraction/guidelines/) for extraction quality.

This tutorial is meant to be simple, but generally should really include reference examples to squeeze out performance!

Using PydanticOutputParser[​](#using-pydanticoutputparser "Direct link to Using PydanticOutputParser")
------------------------------------------------------------------------------------------------------

The following example uses the built-in `PydanticOutputParser` to parse the output of a chat model.

```
from typing import List, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator


class Person(BaseModel):
    """Information about a person."""

    name: str = Field(..., description="The name of the person")
    height_in_meters: float = Field(
        ..., description="The height of the person expressed in meters."
    )


class People(BaseModel):
    """Identifying information about all people in a text."""

    people: List[Person]


# Set up a parser
parser = PydanticOutputParser(pydantic_object=People)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user query. Wrap the output in `json` tags\n{format_instructions}",
        ),
        ("human", "{query}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

```


Let’s take a look at what information is sent to the model

```
query = "Anna is 23 years old and she is 6 feet tall"

```


```
print(prompt.format_prompt(query=query).to_string())

```


```
System: Answer the user query. Wrap the output in `json` tags
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"description": "Identifying information about all people in a text.", "properties": {"people": {"title": "People", "type": "array", "items": {"$ref": "#/definitions/Person"}}}, "required": ["people"], "definitions": {"Person": {"title": "Person", "description": "Information about a person.", "type": "object", "properties": {"name": {"title": "Name", "description": "The name of the person", "type": "string"}, "height_in_meters": {"title": "Height In Meters", "description": "The height of the person expressed in meters.", "type": "number"}}, "required": ["name", "height_in_meters"]}}}
```
Human: Anna is 23 years old and she is 6 feet tall

```


```
chain = prompt | model | parser
chain.invoke({"query": query})

```


```
People(people=[Person(name='Anna', height_in_meters=1.83)])

```


Custom Parsing[​](#custom-parsing "Direct link to Custom Parsing")
------------------------------------------------------------------

It’s easy to create a custom prompt and parser with `LangChain` and `LCEL`.

You can use a simple function to parse the output from the model!

```
import json
import re
from typing import List, Optional

from langchain_anthropic.chat_models import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator


class Person(BaseModel):
    """Information about a person."""

    name: str = Field(..., description="The name of the person")
    height_in_meters: float = Field(
        ..., description="The height of the person expressed in meters."
    )


class People(BaseModel):
    """Identifying information about all people in a text."""

    people: List[Person]


# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user query. Output your answer as JSON that  "
            "matches the given schema: ```json\n{schema}\n```. "
            "Make sure to wrap the answer in ```json and ``` tags",
        ),
        ("human", "{query}"),
    ]
).partial(schema=People.schema())


# Custom parser
def extract_json(message: AIMessage) -> List[dict]:
    """Extracts JSON content from a string where JSON is embedded between ```json and ``` tags.

    Parameters:
        text (str): The text containing the JSON content.

    Returns:
        list: A list of extracted JSON strings.
    """
    text = message.content
    # Define the regular expression pattern to match JSON blocks
    pattern = r"```json(.*?)```"

    # Find all non-overlapping matches of the pattern in the string
    matches = re.findall(pattern, text, re.DOTALL)

    # Return the list of matched JSON strings, stripping any leading or trailing whitespace
    try:
        return [json.loads(match.strip()) for match in matches]
    except Exception:
        raise ValueError(f"Failed to parse: {message}")

```


```
query = "Anna is 23 years old and she is 6 feet tall"
print(prompt.format_prompt(query=query).to_string())

```


```
System: Answer the user query. Output your answer as JSON that  matches the given schema: ```json
{'title': 'People', 'description': 'Identifying information about all people in a text.', 'type': 'object', 'properties': {'people': {'title': 'People', 'type': 'array', 'items': {'$ref': '#/definitions/Person'}}}, 'required': ['people'], 'definitions': {'Person': {'title': 'Person', 'description': 'Information about a person.', 'type': 'object', 'properties': {'name': {'title': 'Name', 'description': 'The name of the person', 'type': 'string'}, 'height_in_meters': {'title': 'Height In Meters', 'description': 'The height of the person expressed in meters.', 'type': 'number'}}, 'required': ['name', 'height_in_meters']}}}
```. Make sure to wrap the answer in ```json and ``` tags
Human: Anna is 23 years old and she is 6 feet tall

```


```
chain = prompt | model | extract_json
chain.invoke({"query": query})

```


```
[{'people': [{'name': 'Anna', 'height_in_meters': 1.83}]}]

```


Other Libraries[​](#other-libraries "Direct link to Other Libraries")
---------------------------------------------------------------------

If you’re looking at extracting using a parsing approach, check out the [Kor](https://eyurtsev.github.io/kor/) library. It’s written by one of the `LangChain` maintainers and it helps to craft a prompt that takes examples into account, allows controlling formats (e.g., JSON or CSV) and expresses the schema in TypeScript. It seems to work pretty!

* * *

#### Help us out by providing feedback on this documentation page:

[](https://python.langchain.com/docs/use_cases/extraction/how_to/handle_files/)[](https://python.langchain.com/docs/use_cases/chatbots/)

*   [Using PydanticOutputParser](#using-pydanticoutputparser)
*   [Custom Parsing](#custom-parsing)
*   [Other Libraries](#other-libraries)