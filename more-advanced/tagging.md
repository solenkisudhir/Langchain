# Tagging | 🦜️🔗 LangChain
Let’s see a very straightforward example of how we can use OpenAI tool calling for tagging in LangChain. We’ll use the [`with_structured_output`](https://python.langchain.com/docs/modules/model_io/chat/structured_output/) method supported by OpenAI models:

Let’s specify a Pydantic model with a few properties and their expected type in our schema.

```
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)


class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10"
    )
    language: str = Field(description="The language the text is written in")


# LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125").with_structured_output(
    Classification
)

tagging_chain = tagging_prompt | llm

```


```
Classification(sentiment='positive', aggressiveness=1, language='Spanish')

```


```
{'sentiment': 'negative', 'aggressiveness': 8, 'language': 'Spanish'}

```


As we can see in the examples, it correctly interprets what we want.

The results vary so that we may get, for example, sentiments in different languages (‘positive’, ‘enojado’ etc.).

We will see how to control these results in the next section.

Careful schema definition gives us more control over the model’s output.

Let’s redeclare our Pydantic model to control for each of the previously mentioned aspects using enums:

```
class Classification(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    aggressiveness: int = Field(
        ...,
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["spanish", "english", "french", "german", "italian"]
    )

```


```
tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125").with_structured_output(
    Classification
)

chain = tagging_prompt | llm

```


```
Classification(sentiment='happy', aggressiveness=1, language='spanish')

```


```
Classification(sentiment='sad', aggressiveness=5, language='spanish')

```


```
inp = "Weather is ok here, I can go outside without much more than a coat"
chain.invoke({"input": inp})

```


```
Classification(sentiment='neutral', aggressiveness=2, language='english')

```
