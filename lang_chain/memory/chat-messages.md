# Chat Messages | 🦜️🔗 LangChain
*   [](https://python.langchain.com/)
*   More
*   [Memory](https://python.langchain.com/docs/modules/memory/)
*   Chat Messages

info

Head to [Integrations](https://python.langchain.com/docs/integrations/memory/) for documentation on built-in memory integrations with 3rd-party databases and tools.

One of the core utility classes underpinning most (if not all) memory modules is the `ChatMessageHistory` class. This is a super lightweight wrapper that provides convenience methods for saving HumanMessages, AIMessages, and then fetching them all.

You may want to use this class directly if you are managing memory outside of a chain.

```
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()

history.add_user_message("hi!")

history.add_ai_message("whats up?")

```


```
    [HumanMessage(content='hi!', additional_kwargs={}),
     AIMessage(content='whats up?', additional_kwargs={})]

```


* * *

#### Help us out by providing feedback on this documentation page:

[

Previous

\[Beta\] Memory

](https://python.langchain.com/docs/modules/memory/)[

Next

Memory in LLMChain

](https://python.langchain.com/docs/modules/memory/adding_memory/)