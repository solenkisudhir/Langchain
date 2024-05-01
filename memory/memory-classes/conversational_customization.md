# Customizing Conversational Memory | 🦜️🔗 LangChain
This notebook walks through a few ways to customize conversational memory.

```
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)

```


AI prefix[​](#ai-prefix "Direct link to AI prefix")
---------------------------------------------------

The first way to do so is by changing the AI prefix in the conversation summary. By default, this is set to “AI”, but you can set this to be anything you want. Note that if you change this, you should also change the prompt used in the chain to reflect this naming change. Let’s walk through an example of that in the example below.

```
# Here it is by default set to "AI"
conversation = ConversationChain(
    llm=llm, verbose=True, memory=ConversationBufferMemory()
)

```


```
conversation.predict(input="Hi there!")

```


```


> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi there!
AI:

> Finished ConversationChain chain.

```


```
" Hi there! It's nice to meet you. How can I help you today?"

```


```
conversation.predict(input="What's the weather?")

```


```


> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi there!
AI:  Hi there! It's nice to meet you. How can I help you today?
Human: What's the weather?
AI:

> Finished ConversationChain chain.

```


```
' The current weather is sunny and warm with a temperature of 75 degrees Fahrenheit. The forecast for the next few days is sunny with temperatures in the mid-70s.'

```


```
# Now we can override it and set it to "AI Assistant"
from langchain_core.prompts.prompt import PromptTemplate

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI Assistant:"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(ai_prefix="AI Assistant"),
)

```


```
conversation.predict(input="Hi there!")

```


```


> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi there!
AI Assistant:

> Finished ConversationChain chain.

```


```
" Hi there! It's nice to meet you. How can I help you today?"

```


```
conversation.predict(input="What's the weather?")

```


```


> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi there!
AI Assistant:  Hi there! It's nice to meet you. How can I help you today?
Human: What's the weather?
AI Assistant:

> Finished ConversationChain chain.

```


```
' The current weather is sunny and warm with a temperature of 75 degrees Fahrenheit. The forecast for the rest of the day is sunny with a high of 78 degrees and a low of 65 degrees.'

```


Human prefix[​](#human-prefix "Direct link to Human prefix")
------------------------------------------------------------

The next way to do so is by changing the Human prefix in the conversation summary. By default, this is set to “Human”, but you can set this to be anything you want. Note that if you change this, you should also change the prompt used in the chain to reflect this naming change. Let’s walk through an example of that in the example below.

```
# Now we can override it and set it to "Friend"
from langchain_core.prompts.prompt import PromptTemplate

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Friend: {input}
AI:"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(human_prefix="Friend"),
)

```


```
conversation.predict(input="Hi there!")

```


```


> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Friend: Hi there!
AI:

> Finished ConversationChain chain.

```


```
" Hi there! It's nice to meet you. How can I help you today?"

```


```
conversation.predict(input="What's the weather?")

```


```


> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Friend: Hi there!
AI:  Hi there! It's nice to meet you. How can I help you today?
Friend: What's the weather?
AI:

> Finished ConversationChain chain.

```


```
' The weather right now is sunny and warm with a temperature of 75 degrees Fahrenheit. The forecast for the rest of the day is mostly sunny with a high of 82 degrees.'

```
