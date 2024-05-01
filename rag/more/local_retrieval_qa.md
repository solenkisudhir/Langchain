# Quick reference | 🦜️🔗 LangChain
Prompt templates are predefined recipes for generating prompts for language models.

A template may include instructions, few-shot examples, and specific context and questions appropriate for a given task.

LangChain provides tooling to create and work with prompt templates.

LangChain strives to create model agnostic templates to make it easy to reuse existing templates across different language models.

Typically, language models expect the prompt to either be a string or else a list of chat messages.

`PromptTemplate`[​](#prompttemplate "Direct link to prompttemplate")
--------------------------------------------------------------------

Use `PromptTemplate` to create a template for a string prompt.

By default, `PromptTemplate` uses [Python’s str.format](https://docs.python.org/3/library/stdtypes.html#str.format) syntax for templating.

```
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)
prompt_template.format(adjective="funny", content="chickens")

```


```
'Tell me a funny joke about chickens.'

```


The template supports any number of variables, including no variables:

```
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me a joke")
prompt_template.format()

```


You can create custom prompt templates that format the prompt in any way you want. For more information, see [Prompt Template Composition](https://python.langchain.com/docs/modules/model_io/prompts/composition/).

`ChatPromptTemplate`[​](#chatprompttemplate "Direct link to chatprompttemplate")
--------------------------------------------------------------------------------

The prompt to [chat models](https://python.langchain.com/docs/modules/model_io/chat/)/ is a list of [chat messages](https://python.langchain.com/docs/modules/model_io/chat/message_types/).

Each chat message is associated with content, and an additional parameter called `role`. For example, in the OpenAI [Chat Completions API](https://platform.openai.com/docs/guides/chat/introduction), a chat message can be associated with an AI assistant, a human or a system role.

Create a chat prompt template like this:

```
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ]
)

messages = chat_template.format_messages(name="Bob", user_input="What is your name?")

```


Piping these formatted messages into LangChain’s `ChatOpenAI` chat model class is roughly equivalent to the following with using the OpenAI client directly:

```
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful AI bot. Your name is Bob."},
        {"role": "user", "content": "Hello, how are you doing?"},
        {"role": "assistant", "content": "I'm doing well, thanks!"},
        {"role": "user", "content": "What is your name?"},
    ],
)

```


The `ChatPromptTemplate.from_messages` static method accepts a variety of message representations and is a convenient way to format input to chat models with exactly the messages you want.

For example, in addition to using the 2-tuple representation of (type, content) used above, you could pass in an instance of `MessagePromptTemplate` or `BaseMessage`.

```
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful assistant that re-writes the user's text to "
                "sound more upbeat."
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)
messages = chat_template.format_messages(text="I don't like eating tasty things")
print(messages)

```


```
[SystemMessage(content="You are a helpful assistant that re-writes the user's text to sound more upbeat."), HumanMessage(content="I don't like eating tasty things")]

```


This provides you with a lot of flexibility in how you construct your chat prompts.

Message Prompts[​](#message-prompts "Direct link to Message Prompts")
---------------------------------------------------------------------

LangChain provides different types of `MessagePromptTemplate`. The most commonly used are `AIMessagePromptTemplate`, `SystemMessagePromptTemplate` and `HumanMessagePromptTemplate`, which create an AI message, system message and human message respectively. You can read more about the [different types of messages here](https://python.langchain.com/docs/modules/model_io/chat/message_types/).

In cases where the chat model supports taking chat message with arbitrary role, you can use `ChatMessagePromptTemplate`, which allows user to specify the role name.

```
from langchain_core.prompts import ChatMessagePromptTemplate

prompt = "May the {subject} be with you"

chat_message_prompt = ChatMessagePromptTemplate.from_template(
    role="Jedi", template=prompt
)
chat_message_prompt.format(subject="force")

```


```
ChatMessage(content='May the force be with you', role='Jedi')

```


`MessagesPlaceholder`[​](#messagesplaceholder "Direct link to messagesplaceholder")
-----------------------------------------------------------------------------------

LangChain also provides `MessagesPlaceholder`, which gives you full control of what messages to be rendered during formatting. This can be useful when you are uncertain of what role you should be using for your message prompt templates or when you wish to insert a list of messages during formatting.

```
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

human_prompt = "Summarize our conversation so far in {word_count} words."
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name="conversation"), human_message_template]
)

```


```
from langchain_core.messages import AIMessage, HumanMessage

human_message = HumanMessage(content="What is the best way to learn programming?")
ai_message = AIMessage(
    content="""\
1. Choose a programming language: Decide on a programming language that you want to learn.

2. Start with the basics: Familiarize yourself with the basic programming concepts such as variables, data types and control structures.

3. Practice, practice, practice: The best way to learn programming is through hands-on experience\
"""
)

chat_prompt.format_prompt(
    conversation=[human_message, ai_message], word_count="10"
).to_messages()

```


```
[HumanMessage(content='What is the best way to learn programming?'),
 AIMessage(content='1. Choose a programming language: Decide on a programming language that you want to learn.\n\n2. Start with the basics: Familiarize yourself with the basic programming concepts such as variables, data types and control structures.\n\n3. Practice, practice, practice: The best way to learn programming is through hands-on experience'),
 HumanMessage(content='Summarize our conversation so far in 10 words.')]

```


The full list of message prompt template types includes:

*   [AIMessagePromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.AIMessagePromptTemplate.html), for AI assistant messages;
*   [SystemMessagePromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.SystemMessagePromptTemplate.html), for system messages;
*   [HumanMessagePromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.HumanMessagePromptTemplate.html), for user messages;
*   [ChatMessagePromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatMessagePromptTemplate.html), for messages with arbitrary roles;
*   [MessagesPlaceholder](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.MessagesPlaceholder.html), which accommodates a list of messages.

LCEL[​](#lcel "Direct link to LCEL")
------------------------------------

`PromptTemplate` and `ChatPromptTemplate` implement the [Runnable interface](https://python.langchain.com/docs/expression_language/interface/), the basic building block of the [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/). This means they support `invoke`, `ainvoke`, `stream`, `astream`, `batch`, `abatch`, `astream_log` calls.

`PromptTemplate` accepts a dictionary (of the prompt variables) and returns a `StringPromptValue`. A `ChatPromptTemplate` accepts a dictionary and returns a `ChatPromptValue`.

```
prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)

prompt_val = prompt_template.invoke({"adjective": "funny", "content": "chickens"})
prompt_val

```


```
StringPromptValue(text='Tell me a funny joke about chickens.')

```


```
'Tell me a funny joke about chickens.'

```


```
[HumanMessage(content='Tell me a funny joke about chickens.')]

```


```
chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful assistant that re-writes the user's text to "
                "sound more upbeat."
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)

chat_val = chat_template.invoke({"text": "i dont like eating tasty things."})

```


```
[SystemMessage(content="You are a helpful assistant that re-writes the user's text to sound more upbeat."),
 HumanMessage(content='i dont like eating tasty things.')]

```


```
"System: You are a helpful assistant that re-writes the user's text to sound more upbeat.\nHuman: i dont like eating tasty things."

```
