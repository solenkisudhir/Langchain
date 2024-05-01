# Retry parser | 🦜️🔗 LangChain
While in some cases it is possible to fix any parsing mistakes by only looking at the output, in other cases it isn’t. An example of this is when the output is not just in the incorrect format, but is partially complete. Consider the below example.

```
from langchain.output_parsers import (
    OutputFixingParser,
    PydanticOutputParser,
)
from langchain_core.prompts import (
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAI

```


```
template = """Based on the user question, provide an Action and Action Input for what step should be taken.
{format_instructions}
Question: {query}
Response:"""


class Action(BaseModel):
    action: str = Field(description="action to take")
    action_input: str = Field(description="input to the action")


parser = PydanticOutputParser(pydantic_object=Action)

```


```
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

```


```
prompt_value = prompt.format_prompt(query="who is leo di caprios gf?")

```


```
bad_response = '{"action": "search"}'

```


If we try to parse this response as is, we will get an error:

```
parser.parse(bad_response)

```


```
OutputParserException: Failed to parse Action from completion {"action": "search"}. Got: 1 validation error for Action
action_input
  field required (type=value_error.missing)

```


If we try to use the `OutputFixingParser` to fix this error, it will be confused - namely, it doesn’t know what to actually put for action input.

```
fix_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())

```


```
fix_parser.parse(bad_response)

```


```
Action(action='search', action_input='input')

```


Instead, we can use the RetryOutputParser, which passes in the prompt (as well as the original output) to try again to get a better response.

```
from langchain.output_parsers import RetryOutputParser

```


```
retry_parser = RetryOutputParser.from_llm(parser=parser, llm=OpenAI(temperature=0))

```


```
retry_parser.parse_with_prompt(bad_response, prompt_value)

```


```
Action(action='search', action_input='leo di caprio girlfriend')

```


We can also add the RetryOutputParser easily with a custom chain which transform the raw LLM/ChatModel output into a more workable format.

```
from langchain_core.runnables import RunnableLambda, RunnableParallel

completion_chain = prompt | OpenAI(temperature=0)

main_chain = RunnableParallel(
    completion=completion_chain, prompt_value=prompt
) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))


main_chain.invoke({"query": "who is leo di caprios gf?"})

```


```
Action(action='search', action_input='leo di caprio girlfriend')

```


Find out api documentation for [RetryOutputParser](https://api.python.langchain.com/en/latest/output_parsers/langchain.output_parsers.retry.RetryOutputParser.html#langchain.output_parsers.retry.RetryOutputParser).