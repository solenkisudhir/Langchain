# CSV parser | 🦜️🔗 LangChain
This output parser can be used when you want to return a list of comma-separated items.

```
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

model = ChatOpenAI(temperature=0)

chain = prompt | model | output_parser

```


```
chain.invoke({"subject": "ice cream flavors"})

```


```
['Vanilla',
 'Chocolate',
 'Strawberry',
 'Mint Chocolate Chip',
 'Cookies and Cream']

```


```
for s in chain.stream({"subject": "ice cream flavors"}):
    print(s)

```


```
['Vanilla']
['Chocolate']
['Strawberry']
['Mint Chocolate Chip']
['Cookies and Cream']

```


Find out api documentation for [CommaSeparatedListOutputParser](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.list.CommaSeparatedListOutputParser.html#langchain_core.output_parsers.list.CommaSeparatedListOutputParser).

* * *

#### Help us out by providing feedback on this documentation page: