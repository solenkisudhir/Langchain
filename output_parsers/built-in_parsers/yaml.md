# YAML parser | 🦜️🔗 LangChain
This output parser allows users to specify an arbitrary schema and query LLMs for outputs that conform to that schema, using YAML to format their response.

Keep in mind that large language models are leaky abstractions! You’ll have to use an LLM with sufficient capacity to generate well-formed YAML. In the OpenAI family, DaVinci can do reliably but Curie’s ability already drops off dramatically.

You can optionally use Pydantic to declare your data model.

```
from typing import List

from langchain.output_parsers import YamlOutputParser
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
parser = YamlOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

chain.invoke({"query": joke_query})

```


```
Joke(setup="Why don't scientists trust atoms?", punchline='Because they make up everything!')

```


Find out api documentation for [YamlOutputParser](https://api.python.langchain.com/en/latest/output_parsers/langchain.output_parsers.yaml.YamlOutputParser.html#langchain.output_parsers.yaml.YamlOutputParser).