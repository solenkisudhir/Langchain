# Enum parser | 🦜️🔗 LangChain
This notebook shows how to use an Enum output parser.

```
from langchain.output_parsers.enum import EnumOutputParser

```


```
from enum import Enum


class Colors(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

```


```
parser = EnumOutputParser(enum=Colors)

```


```
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

prompt = PromptTemplate.from_template(
    """What color eyes does this person have?

> Person: {person}

Instructions: {instructions}"""
).partial(instructions=parser.get_format_instructions())
chain = prompt | ChatOpenAI() | parser

```


```
chain.invoke({"person": "Frank Sinatra"})

```


Find out api documentation for [EnumOutputParser](https://api.python.langchain.com/en/latest/output_parsers/langchain.output_parsers.enum.EnumOutputParser.html#langchain.output_parsers.enum.EnumOutputParser).

* * *

#### Help us out by providing feedback on this documentation page: