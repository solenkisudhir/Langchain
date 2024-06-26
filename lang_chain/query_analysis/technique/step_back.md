# Step Back Prompting | 🦜️🔗 LangChain
Sometimes search quality and model generations can be tripped up by the specifics of a question. One way to handle this is to first generate a more abstract, “step back” question and to query based on both the original and step back question.

For example, if we ask a question of the form “Why does my LangGraph agent astream\_events return {LONG\_TRACE} instead of {DESIRED\_OUTPUT}” we will likely retrieve more relevant documents if we search with the more generic question “How does astream\_events work with a LangGraph agent” than if we search with the specific user question.

Let’s take a look at how we might use step back prompting in the context of our Q&A bot over the LangChain YouTube videos.

Setup[​](#setup "Direct link to Setup")
---------------------------------------

#### Install dependencies[​](#install-dependencies "Direct link to Install dependencies")

```
# %pip install -qU langchain-core langchain-openai

```


#### Set environment variables[​](#set-environment-variables "Direct link to Set environment variables")

We’ll use OpenAI in this example:

```
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

# Optional, uncomment to trace runs with LangSmith. Sign up here: https://smith.langchain.com.
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

```


Step back question generation[​](#step-back-question-generation "Direct link to Step back question generation")
---------------------------------------------------------------------------------------------------------------

Generating good step back questions comes down to writing a good prompt:

```
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

system = """You are an expert at taking a specific question and extracting a more generic question that gets at \
the underlying principles needed to answer the specific question.

You will be asked about a set of software for building LLM-powered applications called LangChain, LangGraph, LangServe, and LangSmith.

LangChain is a Python framework that provides a large set of integrations that can easily be composed to build LLM applications.
LangGraph is a Python package built on top of LangChain that makes it easy to build stateful, multi-actor LLM applications.
LangServe is a Python package built on top of LangChain that makes it easy to deploy a LangChain application as a REST API.
LangSmith is a platform that makes it easy to trace and test LLM applications.

Given a specific user question about one or more of these products, write a more generic question that needs to be answered in order to answer the specific question. \

If you don't recognize a word or acronym to not try to rewrite it.

Write concise questions."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
step_back = prompt | llm | StrOutputParser()

```


```
question = (
    "I built a LangGraph agent using Gemini Pro and tools like vectorstores and duckduckgo search. "
    "How do I get just the LLM calls from the event stream"
)
result = step_back.invoke({"question": question})
print(result)

```


```
What are the specific methods or functions provided by LangGraph for extracting LLM calls from an event stream that includes various types of interactions and data sources?

```


Returning the stepback question and the original question[​](#returning-the-stepback-question-and-the-original-question "Direct link to Returning the stepback question and the original question")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

To increase our recall we’ll likely want to retrieve documents based on both the step back question and the original question. We can easily return both like so:

```
from langchain_core.runnables import RunnablePassthrough

step_back_and_original = RunnablePassthrough.assign(step_back=step_back)

step_back_and_original.invoke({"question": question})

```


```
{'question': 'I built a LangGraph agent using Gemini Pro and tools like vectorstores and duckduckgo search. How do I get just the LLM calls from the event stream',
 'step_back': 'What are the specific methods or functions provided by LangGraph for extracting LLM calls from an event stream generated by an agent built using external tools like Gemini Pro, vectorstores, and DuckDuckGo search?'}

```


Using function-calling to get structured output[​](#using-function-calling-to-get-structured-output "Direct link to Using function-calling to get structured output")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------

If we were composing this technique with other query analysis techniques, we’d likely be using function calling to get out structured query objects. We can use function-calling for step back prompting like so:

```
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel, Field


class StepBackQuery(BaseModel):
    step_back_question: str = Field(
        ...,
        description="Given a specific user question about one or more of these products, write a more generic question that needs to be answered in order to answer the specific question.",
    )


llm_with_tools = llm.bind_tools([StepBackQuery])
hyde_chain = prompt | llm_with_tools | PydanticToolsParser(tools=[StepBackQuery])
hyde_chain.invoke({"question": question})

```


```
[StepBackQuery(step_back_question='What are the steps to filter and extract specific types of calls from an event stream in a Python framework like LangGraph?')]

```


* * *

#### Help us out by providing feedback on this documentation page: