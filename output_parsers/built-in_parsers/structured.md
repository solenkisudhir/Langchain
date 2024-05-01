# Structured output parser | 🦜️🔗 LangChain
This output parser can be used when you want to return multiple fields. While the Pydantic/JSON parser is more powerful, this is useful for less powerful models.

```
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

```


```
response_schemas = [
    ResponseSchema(name="answer", description="answer to the user's question"),
    ResponseSchema(
        name="source",
        description="source used to answer the user's question, should be a website.",
    ),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

```


We now get a string that contains instructions for how the response should be formatted, and we then insert that into our prompt.

```
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="answer the users question as best as possible.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions},
)

```


```
model = ChatOpenAI(temperature=0)
chain = prompt | model | output_parser

```


```
chain.invoke({"question": "what's the capital of france?"})

```


```
{'answer': 'The capital of France is Paris.',
 'source': 'https://en.wikipedia.org/wiki/Paris'}

```


```
for s in chain.stream({"question": "what's the capital of france?"}):
    print(s)

```


```
{'answer': 'The capital of France is Paris.', 'source': 'https://en.wikipedia.org/wiki/Paris'}

```


Find out api documentation for [StructuredOutputParser](https://api.python.langchain.com/en/latest/output_parsers/langchain.output_parsers.structured.StructuredOutputParser.html#langchain.output_parsers.structured.StructuredOutputParser).

* * *

#### Help us out by providing feedback on this documentation page: