# Datetime parser | 🦜️🔗 LangChain
This OutputParser can be used to parse LLM output into datetime format.

```
from langchain.output_parsers import DatetimeOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

```


```
output_parser = DatetimeOutputParser()
template = """Answer the users question:

{question}

{format_instructions}"""
prompt = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

```


```
PromptTemplate(input_variables=['question'], partial_variables={'format_instructions': "Write a datetime string that matches the following pattern: '%Y-%m-%dT%H:%M:%S.%fZ'.\n\nExamples: 0668-08-09T12:56:32.732651Z, 1213-06-23T21:01:36.868629Z, 0713-07-06T18:19:02.257488Z\n\nReturn ONLY this string, no other words!"}, template='Answer the users question:\n\n{question}\n\n{format_instructions}')

```


```
chain = prompt | OpenAI() | output_parser

```


```
output = chain.invoke({"question": "when was bitcoin founded?"})

```


Find out api documentation for [DatetimeOutputParser](https://api.python.langchain.com/en/latest/output_parsers/langchain.output_parsers.datetime.DatetimeOutputParser.html#langchain.output_parsers.datetime.DatetimeOutputParser).

* * *

#### Help us out by providing feedback on this documentation page: