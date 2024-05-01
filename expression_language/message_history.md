# Add message history (memory) | 🦜️🔗 LangChain
The `RunnableWithMessageHistory` lets us add message history to certain types of chains. It wraps another Runnable and manages the chat message history for it.

Specifically, it can be used for any Runnable that takes as input one of

*   a sequence of `BaseMessage`
*   a dict with a key that takes a sequence of `BaseMessage`
*   a dict with a key that takes the latest message(s) as a string or sequence of `BaseMessage`, and a separate key that takes historical messages

And returns as output one of

*   a string that can be treated as the contents of an `AIMessage`
*   a sequence of `BaseMessage`
*   a dict with a key that contains a sequence of `BaseMessage`

Let’s take a look at some examples to see how it works. First we construct a runnable (which here accepts a dict as input and returns a message as output):

```
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're an assistant who's good at {ability}. Respond in 20 words or fewer",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
runnable = prompt | model

```


To manage the message history, we will need: 1. This runnable; 2. A callable that returns an instance of `BaseChatMessageHistory`.

Check out the [memory integrations](https://integrations.langchain.com/memory) page for implementations of chat message histories using Redis and other providers. Here we demonstrate using an in-memory `ChatMessageHistory` as well as more persistent storage using `RedisChatMessageHistory`.

In-memory[​](#in-memory "Direct link to In-memory")
---------------------------------------------------

Below we show a simple example in which the chat history lives in memory, in this case via a global Python dict.

We construct a callable `get_session_history` that references this dict to return an instance of `ChatMessageHistory`. The arguments to the callable can be specified by passing a configuration to the `RunnableWithMessageHistory` at runtime. By default, the configuration parameter is expected to be a single string `session_id`. This can be adjusted via the `history_factory_config` kwarg.

Using the single-parameter default:

```
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

```


Note that we’ve specified `input_messages_key` (the key to be treated as the latest input message) and `history_messages_key` (the key to add historical messages to).

When invoking this new runnable, we specify the corresponding chat history via a configuration parameter:

```
with_message_history.invoke(
    {"ability": "math", "input": "What does cosine mean?"},
    config={"configurable": {"session_id": "abc123"}},
)

```


```
AIMessage(content='Cosine is a trigonometric function that calculates the ratio of the adjacent side to the hypotenuse of a right triangle.')

```


```
# Remembers
with_message_history.invoke(
    {"ability": "math", "input": "What?"},
    config={"configurable": {"session_id": "abc123"}},
)

```


```
AIMessage(content='Cosine is a mathematical function used to calculate the length of a side in a right triangle.')

```


```
# New session_id --> does not remember.
with_message_history.invoke(
    {"ability": "math", "input": "What?"},
    config={"configurable": {"session_id": "def234"}},
)

```


```
AIMessage(content='I can help with math problems. What do you need assistance with?')

```


The configuration parameters by which we track message histories can be customized by passing in a list of `ConfigurableFieldSpec` objects to the `history_factory_config` parameter. Below, we use two parameters: a `user_id` and `conversation_id`.

```
from langchain_core.runnables import ConfigurableFieldSpec

store = {}


def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]


with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
)

```


```
with_message_history.invoke(
    {"ability": "math", "input": "Hello"},
    config={"configurable": {"user_id": "123", "conversation_id": "1"}},
)

```


### Examples with runnables of different signatures[​](#examples-with-runnables-of-different-signatures "Direct link to Examples with runnables of different signatures")

The above runnable takes a dict as input and returns a BaseMessage. Below we show some alternatives.

#### Messages input, dict output[​](#messages-input-dict-output "Direct link to Messages input, dict output")

```
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel({"output_message": ChatOpenAI()})


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    output_messages_key="output_message",
)

with_message_history.invoke(
    [HumanMessage(content="What did Simone de Beauvoir believe about free will")],
    config={"configurable": {"session_id": "baz"}},
)

```


```
{'output_message': AIMessage(content="Simone de Beauvoir believed in the existence of free will. She argued that individuals have the ability to make choices and determine their own actions, even in the face of social and cultural constraints. She rejected the idea that individuals are purely products of their environment or predetermined by biology or destiny. Instead, she emphasized the importance of personal responsibility and the need for individuals to actively engage in creating their own lives and defining their own existence. De Beauvoir believed that freedom and agency come from recognizing one's own freedom and actively exercising it in the pursuit of personal and collective liberation.")}

```


```
with_message_history.invoke(
    [HumanMessage(content="How did this compare to Sartre")],
    config={"configurable": {"session_id": "baz"}},
)

```


```
{'output_message': AIMessage(content='Simone de Beauvoir\'s views on free will were closely aligned with those of her contemporary and partner Jean-Paul Sartre. Both de Beauvoir and Sartre were existentialist philosophers who emphasized the importance of individual freedom and the rejection of determinism. They believed that human beings have the capacity to transcend their circumstances and create their own meaning and values.\n\nSartre, in his famous work "Being and Nothingness," argued that human beings are condemned to be free, meaning that we are burdened with the responsibility of making choices and defining ourselves in a world that lacks inherent meaning. Like de Beauvoir, Sartre believed that individuals have the ability to exercise their freedom and make choices in the face of external and internal constraints.\n\nWhile there may be some nuanced differences in their philosophical writings, overall, de Beauvoir and Sartre shared a similar belief in the existence of free will and the importance of individual agency in shaping one\'s own life.')}

```


#### Messages input, messages output[​](#messages-input-messages-output "Direct link to Messages input, messages output")

```
RunnableWithMessageHistory(
    ChatOpenAI(),
    get_session_history,
)

```


#### Dict with single key for all messages input, messages output[​](#dict-with-single-key-for-all-messages-input-messages-output "Direct link to Dict with single key for all messages input, messages output")

```
from operator import itemgetter

RunnableWithMessageHistory(
    itemgetter("input_messages") | ChatOpenAI(),
    get_session_history,
    input_messages_key="input_messages",
)

```


Persistent storage[​](#persistent-storage "Direct link to Persistent storage")
------------------------------------------------------------------------------

In many cases it is preferable to persist conversation histories. `RunnableWithMessageHistory` is agnostic as to how the `get_session_history` callable retrieves its chat message histories. See [here](https://github.com/langchain-ai/langserve/blob/main/examples/chat_with_persistence_and_user/server.py) for an example using a local filesystem. Below we demonstrate how one could use Redis. Check out the [memory integrations](https://integrations.langchain.com/memory) page for implementations of chat message histories using other providers.

### Setup[​](#setup "Direct link to Setup")

We’ll need to install Redis if it’s not installed already:

```
%pip install --upgrade --quiet redis

```


Start a local Redis Stack server if we don’t have an existing Redis deployment to connect to:

```
docker run -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

```


```
REDIS_URL = "redis://localhost:6379/0"

```


### [LangSmith](https://python.langchain.com/docs/langsmith/)[​](#langsmith "Direct link to langsmith")

LangSmith is especially useful for something like message history injection, where it can be hard to otherwise understand what the inputs are to various parts of the chain.

Note that LangSmith is not needed, but it is helpful. If you do want to use LangSmith, after you sign up at the link above, make sure to uncoment the below and set your environment variables to start logging traces:

```
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

```


Updating the message history implementation just requires us to define a new callable, this time returning an instance of `RedisChatMessageHistory`:

```
from langchain_community.chat_message_histories import RedisChatMessageHistory


def get_message_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(session_id, url=REDIS_URL)


with_message_history = RunnableWithMessageHistory(
    runnable,
    get_message_history,
    input_messages_key="input",
    history_messages_key="history",
)

```


We can invoke as before:

```
with_message_history.invoke(
    {"ability": "math", "input": "What does cosine mean?"},
    config={"configurable": {"session_id": "foobar"}},
)

```


```
AIMessage(content='Cosine is a trigonometric function that represents the ratio of the adjacent side to the hypotenuse in a right triangle.')

```


```
with_message_history.invoke(
    {"ability": "math", "input": "What's its inverse"},
    config={"configurable": {"session_id": "foobar"}},
)

```


```
AIMessage(content='The inverse of cosine is the arccosine function, denoted as acos or cos^-1, which gives the angle corresponding to a given cosine value.')

```


Looking at the Langsmith trace for the second call, we can see that when constructing the prompt, a “history” variable has been injected which is a list of two messages (our first input and first output).