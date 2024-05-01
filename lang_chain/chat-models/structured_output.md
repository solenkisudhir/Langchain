# Structured Output | 🦜️🔗 LangChain
It is often crucial to have LLMs return structured output. This is because oftentimes the outputs of the LLMs are used in downstream applications, where specific arguments are required. Having the LLM return structured output reliably is necessary for that.

There are a few different high level strategies that are used to do this:

*   Prompting: This is when you ask the LLM (very nicely) to return output in the desired format (JSON, XML). This is nice because it works with all LLMs. It is not nice because there is no guarantee that the LLM returns the output in the right format.
*   Function calling: This is when the LLM is fine-tuned to be able to not just generate a completion, but also generate a function call. The functions the LLM can call are generally passed as extra parameters to the model API. The function names and descriptions should be treated as part of the prompt (they usually count against token counts, and are used by the LLM to decide what to do).
*   Tool calling: A technique similar to function calling, but it allows the LLM to call multiple functions at the same time.
*   JSON mode: This is when the LLM is guaranteed to return JSON.

Different models may support different variants of these, with slightly different parameters. In order to make it easy to get LLMs to return structured output, we have added a common interface to LangChain models: `.with_structured_output`.

By invoking this method (and passing in a JSON schema or a Pydantic model) the model will add whatever model parameters + output parsers are necessary to get back the structured output. There may be more than one way to do this (e.g., function calling vs JSON mode) - you can configure which method to use by passing into that method.

Let’s look at some examples of this in action!

We will use Pydantic to easily structure the response schema.

```
from langchain_core.pydantic_v1 import BaseModel, Field


class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")

```


OpenAI[​](#openai "Direct link to OpenAI")
------------------------------------------

OpenAI exposes a few different ways to get structured outputs.

[API reference](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html#langchain_openai.chat_models.base.ChatOpenAI.with_structured_output)

```
from langchain_openai import ChatOpenAI

```


#### Tool/function Calling[​](#toolfunction-calling "Direct link to Tool/function Calling")

By default, we will use `function_calling`

```
model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = model.with_structured_output(Joke)

```


```
structured_llm.invoke("Tell me a joke about cats")

```


```
Joke(setup='Why was the cat sitting on the computer?', punchline='To keep an eye on the mouse!')

```


#### JSON Mode[​](#json-mode "Direct link to JSON Mode")

We also support JSON mode. Note that we need to specify in the prompt the format that it should respond in.

```
structured_llm = model.with_structured_output(Joke, method="json_mode")

```


```
structured_llm.invoke(
    "Tell me a joke about cats, respond in JSON with `setup` and `punchline` keys"
)

```


```
Joke(setup='Why was the cat sitting on the computer?', punchline='Because it wanted to keep an eye on the mouse!')

```


Fireworks[​](#fireworks "Direct link to Fireworks")
---------------------------------------------------

[Fireworks](https://fireworks.ai/) similarly supports function calling and JSON mode for select models.

[API reference](https://api.python.langchain.com/en/latest/chat_models/langchain_fireworks.chat_models.ChatFireworks.html#langchain_fireworks.chat_models.ChatFireworks.with_structured_output)

```
from langchain_fireworks import ChatFireworks

```


#### Tool/function Calling[​](#toolfunction-calling-1 "Direct link to Tool/function Calling")

By default, we will use `function_calling`

```
model = ChatFireworks(model="accounts/fireworks/models/firefunction-v1")
structured_llm = model.with_structured_output(Joke)

```


```
structured_llm.invoke("Tell me a joke about cats")

```


```
Joke(setup="Why don't cats play poker in the jungle?", punchline='Too many cheetahs!')

```


#### JSON Mode[​](#json-mode-1 "Direct link to JSON Mode")

We also support JSON mode. Note that we need to specify in the prompt the format that it should respond in.

```
structured_llm = model.with_structured_output(Joke, method="json_mode")

```


```
structured_llm.invoke(
    "Tell me a joke about dogs, respond in JSON with `setup` and `punchline` keys"
)

```


```
Joke(setup='Why did the dog sit in the shade?', punchline='To avoid getting burned.')

```


Mistral[​](#mistral "Direct link to Mistral")
---------------------------------------------

We also support structured output with Mistral models, although we only support function calling.

[API reference](https://api.python.langchain.com/en/latest/chat_models/langchain_mistralai.chat_models.ChatMistralAI.html#langchain_mistralai.chat_models.ChatMistralAI.with_structured_output)

```
from langchain_mistralai import ChatMistralAI

```


```
model = ChatMistralAI(model="mistral-large-latest")
structured_llm = model.with_structured_output(Joke)

```


```
structured_llm.invoke("Tell me a joke about cats")

```


```
Joke(setup="Why don't cats play poker in the jungle?", punchline='Too many cheetahs!')

```


Together[​](#together "Direct link to Together")
------------------------------------------------

Since [TogetherAI](https://www.together.ai/) is just a drop in replacement for OpenAI, we can just use the OpenAI integration

```
import os

from langchain_openai import ChatOpenAI

```


```
model = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
)
structured_llm = model.with_structured_output(Joke)

```


```
structured_llm.invoke("Tell me a joke about cats")

```


```
Joke(setup='Why did the cat sit on the computer?', punchline='To keep an eye on the mouse!')

```


Groq[​](#groq "Direct link to Groq")
------------------------------------

Groq provides an OpenAI-compatible function calling API.

[API reference](https://api.python.langchain.com/en/latest/chat_models/langchain_groq.chat_models.ChatGroq.html#langchain_groq.chat_models.ChatGroq.with_structured_output)

```
from langchain_groq import ChatGroq

```


#### Tool/function Calling[​](#toolfunction-calling-2 "Direct link to Tool/function Calling")

By default, we will use `function_calling`

```
model = ChatGroq()
structured_llm = model.with_structured_output(Joke)

```


```
structured_llm.invoke("Tell me a joke about cats")

```


```
Joke(setup="Why don't cats play poker in the jungle?", punchline='Too many cheetahs!')

```


#### JSON Mode[​](#json-mode-2 "Direct link to JSON Mode")

We also support JSON mode. Note that we need to specify in the prompt the format that it should respond in.

```
structured_llm = model.with_structured_output(Joke, method="json_mode")

```


```
structured_llm.invoke(
    "Tell me a joke about cats, respond in JSON with `setup` and `punchline` keys"
)

```


```
Joke(setup="Why don't cats play poker in the jungle?", punchline='Too many cheetahs!')

```


Anthropic[​](#anthropic "Direct link to Anthropic")
---------------------------------------------------

Anthropic’s tool-calling API can be used for structuring outputs. Note that there is currently no way to force a tool-call via the API, so prompting the model correctly is still important.

[API reference](https://api.python.langchain.com/en/latest/chat_models/langchain_anthropic.chat_models.ChatAnthropic.html#langchain_anthropic.chat_models.ChatAnthropic.with_structured_output)

```
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
structured_llm = model.with_structured_output(Joke)
structured_llm.invoke("Tell me a joke about cats. Make sure to call the Joke function.")

```


```
Joke(setup='What do you call a cat that loves to bowl?', punchline='An alley cat!')

```


Google Vertex AI[​](#google-vertex-ai "Direct link to Google Vertex AI")
------------------------------------------------------------------------

Google’s Gemini models support [function-calling](https://ai.google.dev/docs/function_calling), which we can access via Vertex AI and use for structuring outputs.

[API reference](https://api.python.langchain.com/en/latest/chat_models/langchain_google_vertexai.chat_models.ChatVertexAI.html#langchain_google_vertexai.chat_models.ChatVertexAI.with_structured_output)

```
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model="gemini-pro", temperature=0)
structured_llm = llm.with_structured_output(Joke)
structured_llm.invoke("Tell me a joke about cats")

```


```
Joke(setup='Why did the scarecrow win an award?', punchline='Why did the scarecrow win an award? Because he was outstanding in his field.')

```
