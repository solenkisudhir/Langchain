# Memory management | 🦜️🔗 LangChain
A key feature of chatbots is their ability to use content of previous conversation turns as context. This state management can take several forms, including:

*   Simply stuffing previous messages into a chat model prompt.
*   The above, but trimming old messages to reduce the amount of distracting information the model has to deal with.
*   More complex modifications like synthesizing summaries for long running conversations.

We’ll go into more detail on a few techniques below!

Setup[​](#setup "Direct link to Setup")
---------------------------------------

You’ll need to install a few packages, and have your OpenAI API key set as an environment variable named `OPENAI_API_KEY`:

```
%pip install --upgrade --quiet langchain langchain-openai

# Set env var OPENAI_API_KEY or load from a .env file:
import dotenv

dotenv.load_dotenv()

```


```
WARNING: You are using pip version 22.0.4; however, version 23.3.2 is available.
You should consider upgrading via the '/Users/jacoblee/.pyenv/versions/3.10.5/bin/python -m pip install --upgrade pip' command.
Note: you may need to restart the kernel to use updated packages.

```


Let’s also set up a chat model that we’ll use for the below examples.

```
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo-1106")

```


Message passing[​](#message-passing "Direct link to Message passing")
---------------------------------------------------------------------

The simplest form of memory is simply passing chat history messages into a chain. Here’s an example:

```
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | chat

chain.invoke(
    {
        "messages": [
            HumanMessage(
                content="Translate this sentence from English to French: I love programming."
            ),
            AIMessage(content="J'adore la programmation."),
            HumanMessage(content="What did you just say?"),
        ],
    }
)

```


```
AIMessage(content='I said "J\'adore la programmation," which means "I love programming" in French.')

```


We can see that by passing the previous conversation into a chain, it can use it as context to answer questions. This is the basic concept underpinning chatbot memory - the rest of the guide will demonstrate convenient techniques for passing or reformatting messages.

Chat history[​](#chat-history "Direct link to Chat history")
------------------------------------------------------------

It’s perfectly fine to store and pass messages directly as an array, but we can use LangChain’s built-in [message history class](https://python.langchain.com/docs/modules/memory/chat_messages/) to store and load messages as well. Instances of this class are responsible for storing and loading chat messages from persistent storage. LangChain integrates with many providers - you can see a [list of integrations here](https://python.langchain.com/docs/integrations/memory/) - but for this demo we will use an ephemeral demo class.

Here’s an example of the API:

```
from langchain.memory import ChatMessageHistory

demo_ephemeral_chat_history = ChatMessageHistory()

demo_ephemeral_chat_history.add_user_message(
    "Translate this sentence from English to French: I love programming."
)

demo_ephemeral_chat_history.add_ai_message("J'adore la programmation.")

demo_ephemeral_chat_history.messages

```


```
[HumanMessage(content='Translate this sentence from English to French: I love programming.'),
 AIMessage(content="J'adore la programmation.")]

```


We can use it directly to store conversation turns for our chain:

```
demo_ephemeral_chat_history = ChatMessageHistory()

input1 = "Translate this sentence from English to French: I love programming."

demo_ephemeral_chat_history.add_user_message(input1)

response = chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
    }
)

demo_ephemeral_chat_history.add_ai_message(response)

input2 = "What did I just ask you?"

demo_ephemeral_chat_history.add_user_message(input2)

chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
    }
)

```


```
AIMessage(content='You asked me to translate the sentence "I love programming" from English to French.')

```


Automatic history management[​](#automatic-history-management "Direct link to Automatic history management")
------------------------------------------------------------------------------------------------------------

The previous examples pass messages to the chain explicitly. This is a completely acceptable approach, but it does require external management of new messages. LangChain also includes an wrapper for LCEL chains that can handle this process automatically called `RunnableWithMessageHistory`.

To show how it works, let’s slightly modify the above prompt to take a final `input` variable that populates a `HumanMessage` template after the chat history. This means that we will expect a `chat_history` parameter that contains all messages BEFORE the current messages instead of all messages:

```
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

chain = prompt | chat

```


We’ll pass the latest input to the conversation here and let the `RunnableWithMessageHistory` class wrap our chain and do the work of appending that `input` variable to the chat history.

Next, let’s declare our wrapped chain:

```
from langchain_core.runnables.history import RunnableWithMessageHistory

demo_ephemeral_chat_history_for_chain = ChatMessageHistory()

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: demo_ephemeral_chat_history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history",
)

```


This class takes a few parameters in addition to the chain that we want to wrap:

*   A factory function that returns a message history for a given session id. This allows your chain to handle multiple users at once by loading different messages for different conversations.
*   An `input_messages_key` that specifies which part of the input should be tracked and stored in the chat history. In this example, we want to track the string passed in as `input`.
*   A `history_messages_key` that specifies what the previous messages should be injected into the prompt as. Our prompt has a `MessagesPlaceholder` named `chat_history`, so we specify this property to match.
*   (For chains with multiple outputs) an `output_messages_key` which specifies which output to store as history. This is the inverse of `input_messages_key`.

We can invoke this new chain as normal, with an additional `configurable` field that specifies the particular `session_id` to pass to the factory function. This is unused for the demo, but in real-world chains, you’ll want to return a chat history corresponding to the passed session:

```
chain_with_message_history.invoke(
    {"input": "Translate this sentence from English to French: I love programming."},
    {"configurable": {"session_id": "unused"}},
)

```


```
AIMessage(content='The translation of "I love programming" in French is "J\'adore la programmation."')

```


```
chain_with_message_history.invoke(
    {"input": "What did I just ask you?"}, {"configurable": {"session_id": "unused"}}
)

```


```
AIMessage(content='You just asked me to translate the sentence "I love programming" from English to French.')

```


Modifying chat history[​](#modifying-chat-history "Direct link to Modifying chat history")
------------------------------------------------------------------------------------------

Modifying stored chat messages can help your chatbot handle a variety of situations. Here are some examples:

### Trimming messages[​](#trimming-messages "Direct link to Trimming messages")

LLMs and chat models have limited context windows, and even if you’re not directly hitting limits, you may want to limit the amount of distraction the model has to deal with. One solution is to only load and store the most recent `n` messages. Let’s use an example history with some preloaded messages:

```
demo_ephemeral_chat_history = ChatMessageHistory()

demo_ephemeral_chat_history.add_user_message("Hey there! I'm Nemo.")
demo_ephemeral_chat_history.add_ai_message("Hello!")
demo_ephemeral_chat_history.add_user_message("How are you today?")
demo_ephemeral_chat_history.add_ai_message("Fine thanks!")

demo_ephemeral_chat_history.messages

```


```
[HumanMessage(content="Hey there! I'm Nemo."),
 AIMessage(content='Hello!'),
 HumanMessage(content='How are you today?'),
 AIMessage(content='Fine thanks!')]

```


Let’s use this message history with the `RunnableWithMessageHistory` chain we declared above:

```
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

chain = prompt | chat

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: demo_ephemeral_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

chain_with_message_history.invoke(
    {"input": "What's my name?"},
    {"configurable": {"session_id": "unused"}},
)

```


```
AIMessage(content='Your name is Nemo.')

```


We can see the chain remembers the preloaded name.

But let’s say we have a very small context window, and we want to trim the number of messages passed to the chain to only the 2 most recent ones. We can use the `clear` method to remove messages and re-add them to the history. We don’t have to, but let’s put this method at the front of our chain to ensure it’s always called:

```
from langchain_core.runnables import RunnablePassthrough


def trim_messages(chain_input):
    stored_messages = demo_ephemeral_chat_history.messages
    if len(stored_messages) <= 2:
        return False

    demo_ephemeral_chat_history.clear()

    for message in stored_messages[-2:]:
        demo_ephemeral_chat_history.add_message(message)

    return True


chain_with_trimming = (
    RunnablePassthrough.assign(messages_trimmed=trim_messages)
    | chain_with_message_history
)

```


Let’s call this new chain and check the messages afterwards:

```
chain_with_trimming.invoke(
    {"input": "Where does P. Sherman live?"},
    {"configurable": {"session_id": "unused"}},
)

```


```
AIMessage(content="P. Sherman's address is 42 Wallaby Way, Sydney.")

```


```
demo_ephemeral_chat_history.messages

```


```
[HumanMessage(content="What's my name?"),
 AIMessage(content='Your name is Nemo.'),
 HumanMessage(content='Where does P. Sherman live?'),
 AIMessage(content="P. Sherman's address is 42 Wallaby Way, Sydney.")]

```


And we can see that our history has removed the two oldest messages while still adding the most recent conversation at the end. The next time the chain is called, `trim_messages` will be called again, and only the two most recent messages will be passed to the model. In this case, this means that the model will forget the name we gave it the next time we invoke it:

```
chain_with_trimming.invoke(
    {"input": "What is my name?"},
    {"configurable": {"session_id": "unused"}},
)

```


```
AIMessage(content="I'm sorry, I don't have access to your personal information.")

```


```
demo_ephemeral_chat_history.messages

```


```
[HumanMessage(content='Where does P. Sherman live?'),
 AIMessage(content="P. Sherman's address is 42 Wallaby Way, Sydney."),
 HumanMessage(content='What is my name?'),
 AIMessage(content="I'm sorry, I don't have access to your personal information.")]

```


### Summary memory[​](#summary-memory "Direct link to Summary memory")

We can use this same pattern in other ways too. For example, we could use an additional LLM call to generate a summary of the conversation before calling our chain. Let’s recreate our chat history and chatbot chain:

```
demo_ephemeral_chat_history = ChatMessageHistory()

demo_ephemeral_chat_history.add_user_message("Hey there! I'm Nemo.")
demo_ephemeral_chat_history.add_ai_message("Hello!")
demo_ephemeral_chat_history.add_user_message("How are you today?")
demo_ephemeral_chat_history.add_ai_message("Fine thanks!")

demo_ephemeral_chat_history.messages

```


```
[HumanMessage(content="Hey there! I'm Nemo."),
 AIMessage(content='Hello!'),
 HumanMessage(content='How are you today?'),
 AIMessage(content='Fine thanks!')]

```


We’ll slightly modify the prompt to make the LLM aware that will receive a condensed summary instead of a chat history:

```
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability. The provided chat history includes facts about the user you are speaking with.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)

chain = prompt | chat

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: demo_ephemeral_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

```


And now, let’s create a function that will distill previous interactions into a summary. We can add this one to the front of the chain too:

```
def summarize_messages(chain_input):
    stored_messages = demo_ephemeral_chat_history.messages
    if len(stored_messages) == 0:
        return False
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                "Distill the above chat messages into a single summary message. Include as many specific details as you can.",
            ),
        ]
    )
    summarization_chain = summarization_prompt | chat

    summary_message = summarization_chain.invoke({"chat_history": stored_messages})

    demo_ephemeral_chat_history.clear()

    demo_ephemeral_chat_history.add_message(summary_message)

    return True


chain_with_summarization = (
    RunnablePassthrough.assign(messages_summarized=summarize_messages)
    | chain_with_message_history
)

```


Let’s see if it remembers the name we gave it:

```
chain_with_summarization.invoke(
    {"input": "What did I say my name was?"},
    {"configurable": {"session_id": "unused"}},
)

```


```
AIMessage(content='You introduced yourself as Nemo. How can I assist you today, Nemo?')

```


```
demo_ephemeral_chat_history.messages

```


```
[AIMessage(content='The conversation is between Nemo and an AI. Nemo introduces himself and the AI responds with a greeting. Nemo then asks the AI how it is doing, and the AI responds that it is fine.'),
 HumanMessage(content='What did I say my name was?'),
 AIMessage(content='You introduced yourself as Nemo. How can I assist you today, Nemo?')]

```


Note that invoking the chain again will generate another summary generated from the initial summary plus new messages and so on. You could also design a hybrid approach where a certain number of messages are retained in chat history while others are summarized.