# JSON parser | 🦜️🔗 LangChain
This output parser allows users to specify an arbitrary JSON schema and query LLMs for outputs that conform to that schema.

Keep in mind that large language models are leaky abstractions! You’ll have to use an LLM with sufficient capacity to generate well-formed JSON. In the OpenAI family, DaVinci can do reliably but Curie’s ability already drops off dramatically.

You can optionally use Pydantic to declare your data model.

```
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

```


```
model = ChatOpenAI(temperature=0)

```


```
# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

```


```
# And a query intented to prompt a language model to populate the data structure.
joke_query = "Tell me a joke."

# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

chain.invoke({"query": joke_query})

```


```
{'setup': "Why don't scientists trust atoms?",
 'punchline': 'Because they make up everything!'}

```


Streaming[​](#streaming "Direct link to Streaming")
---------------------------------------------------

This output parser supports streaming.

```
for s in chain.stream({"query": joke_query}):
    print(s)

```


```
{'setup': ''}
{'setup': 'Why'}
{'setup': 'Why don'}
{'setup': "Why don't"}
{'setup': "Why don't scientists"}
{'setup': "Why don't scientists trust"}
{'setup': "Why don't scientists trust atoms"}
{'setup': "Why don't scientists trust atoms?", 'punchline': ''}
{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because'}
{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they'}
{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make'}
{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up'}
{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything'}
{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}

```


Without Pydantic[​](#without-pydantic "Direct link to Without Pydantic")
------------------------------------------------------------------------

You can also use this without Pydantic. This will prompt it return JSON, but doesn’t provide specific about what the schema should be.

```
joke_query = "Tell me a joke."

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

chain.invoke({"query": joke_query})

```


```
{'joke': "Why don't scientists trust atoms? Because they make up everything!"}

```


Find out api documentation for [JsonOutputParser](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.json.JsonOutputParser.html#langchain_core.output_parsers.json.JsonOutputParser).