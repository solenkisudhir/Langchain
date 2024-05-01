# Output-fixing parser | 🦜️🔗 LangChain
This output parser wraps another output parser, and in the event that the first one fails it calls out to another LLM to fix any errors.

But we can do other things besides throw errors. Specifically, we can pass the misformatted output, along with the formatted instructions, to the model and ask it to fix it.

For this example, we’ll use the above Pydantic output parser. Here’s what happens if we pass it a result that does not comply with the schema:

```
from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

```


```
class Actor(BaseModel):
    name: str = Field(description="name of an actor")
    film_names: List[str] = Field(description="list of names of films they starred in")


actor_query = "Generate the filmography for a random actor."

parser = PydanticOutputParser(pydantic_object=Actor)

```


```
misformatted = "{'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}"

```


```
parser.parse(misformatted)

```


```
OutputParserException: Failed to parse Actor from completion {'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}. Got: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)

```


Now we can construct and use a `OutputFixingParser`. This output parser takes as an argument another output parser but also an LLM with which to try to correct any formatting mistakes.

```
from langchain.output_parsers import OutputFixingParser

new_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())

```


```
new_parser.parse(misformatted)

```


```
Actor(name='Tom Hanks', film_names=['Forrest Gump'])

```


Find out api documentation for [OutputFixingParser](https://api.python.langchain.com/en/latest/output_parsers/langchain.output_parsers.fix.OutputFixingParser.html#langchain.output_parsers.fix.OutputFixingParser).