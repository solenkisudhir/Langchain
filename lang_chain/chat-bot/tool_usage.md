# Tool usage | 🦜️🔗 LangChain
This section will cover how to create conversational agents: chatbots that can interact with other systems and APIs using tools.

Before reading this guide, we recommend you read both [the chatbot quickstart](https://python.langchain.com/docs/use_cases/chatbots/quickstart/) in this section and be familiar with [the documentation on agents](https://python.langchain.com/docs/modules/agents/).

Setup[​](#setup "Direct link to Setup")
---------------------------------------

For this guide, we’ll be using an [OpenAI tools agent](https://python.langchain.com/docs/modules/agents/agent_types/openai_tools/) with a single tool for searching the web. The default will be powered by [Tavily](https://python.langchain.com/docs/integrations/tools/tavily_search/), but you can switch it out for any similar tool. The rest of this section will assume you’re using Tavily.

You’ll need to [sign up for an account](https://tavily.com/) on the Tavily website, and install the following packages:

```
%pip install --upgrade --quiet langchain-openai tavily-python

# Set env var OPENAI_API_KEY or load from a .env file:
import dotenv

dotenv.load_dotenv()

```


```
WARNING: You are using pip version 22.0.4; however, version 23.3.2 is available.
You should consider upgrading via the '/Users/jacoblee/.pyenv/versions/3.10.5/bin/python -m pip install --upgrade pip' command.
Note: you may need to restart the kernel to use updated packages.

```


You will also need your OpenAI key set as `OPENAI_API_KEY` and your Tavily API key set as `TAVILY_API_KEY`.

Creating an agent[​](#creating-an-agent "Direct link to Creating an agent")
---------------------------------------------------------------------------

Our end goal is to create an agent that can respond conversationally to user questions while looking up information as needed.

First, let’s initialize Tavily and an OpenAI chat model capable of tool calling:

```
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

tools = [TavilySearchResults(max_results=1)]

# Choose the LLM that will drive the agent
# Only certain models support this
chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

```


To make our agent conversational, we must also choose a prompt with a placeholder for our chat history. Here’s an example:

```
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Adapted from https://smith.langchain.com/hub/hwchase17/openai-tools-agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. You may not need to use tools for every query - the user may just want to chat!",
        ),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

```


Great! Now let’s assemble our agent:

```
from langchain.agents import AgentExecutor, create_openai_tools_agent

agent = create_openai_tools_agent(chat, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

```


Running the agent[​](#running-the-agent "Direct link to Running the agent")
---------------------------------------------------------------------------

Now that we’ve set up our agent, let’s try interacting with it! It can handle both trivial queries that require no lookup:

```
from langchain_core.messages import HumanMessage

agent_executor.invoke({"messages": [HumanMessage(content="I'm Nemo!")]})

```


```


> Entering new AgentExecutor chain...
Hello Nemo! It's great to meet you. How can I assist you today?

> Finished chain.

```


```
{'messages': [HumanMessage(content="I'm Nemo!")],
 'output': "Hello Nemo! It's great to meet you. How can I assist you today?"}

```


Or, it can use of the passed search tool to get up to date information if needed:

```
agent_executor.invoke(
    {
        "messages": [
            HumanMessage(
                content="What is the current conservation status of the Great Barrier Reef?"
            )
        ],
    }
)

```


```


> Entering new AgentExecutor chain...

Invoking: `tavily_search_results_json` with `{'query': 'current conservation status of the Great Barrier Reef'}`


[{'url': 'https://www.barrierreef.org/news/blog/this-is-the-critical-decade-for-coral-reef-survival', 'content': "global coral reef conservation.  © 2024 Great Barrier Reef Foundation. Website by bigfish.tv  #Related News · 29 January 2024 290m more baby corals to help restore and protect the Great Barrier Reef  Great Barrier Reef Foundation Managing Director Anna Marsden says it’s not too late if we act now.The Status of Coral Reefs of the World: 2020 report is the largest analysis of global coral reef health ever undertaken. It found that 14 per cent of the world's coral has been lost since 2009. The report also noted, however, that some of these corals recovered during the 10 years to 2019."}]The current conservation status of the Great Barrier Reef is a critical concern. According to the Great Barrier Reef Foundation, the Status of Coral Reefs of the World: 2020 report found that 14% of the world's coral has been lost since 2009. However, the report also noted that some of these corals recovered during the 10 years to 2019. For more information, you can visit the following link: [Great Barrier Reef Foundation - Conservation Status](https://www.barrierreef.org/news/blog/this-is-the-critical-decade-for-coral-reef-survival)

> Finished chain.

```


```
{'messages': [HumanMessage(content='What is the current conservation status of the Great Barrier Reef?')],
 'output': "The current conservation status of the Great Barrier Reef is a critical concern. According to the Great Barrier Reef Foundation, the Status of Coral Reefs of the World: 2020 report found that 14% of the world's coral has been lost since 2009. However, the report also noted that some of these corals recovered during the 10 years to 2019. For more information, you can visit the following link: [Great Barrier Reef Foundation - Conservation Status](https://www.barrierreef.org/news/blog/this-is-the-critical-decade-for-coral-reef-survival)"}

```


Conversational responses[​](#conversational-responses "Direct link to Conversational responses")
------------------------------------------------------------------------------------------------

Because our prompt contains a placeholder for chat history messages, our agent can also take previous interactions into account and respond conversationally like a standard chatbot:

```
from langchain_core.messages import AIMessage, HumanMessage

agent_executor.invoke(
    {
        "messages": [
            HumanMessage(content="I'm Nemo!"),
            AIMessage(content="Hello Nemo! How can I assist you today?"),
            HumanMessage(content="What is my name?"),
        ],
    }
)

```


```


> Entering new AgentExecutor chain...
Your name is Nemo!

> Finished chain.

```


```
{'messages': [HumanMessage(content="I'm Nemo!"),
  AIMessage(content='Hello Nemo! How can I assist you today?'),
  HumanMessage(content='What is my name?')],
 'output': 'Your name is Nemo!'}

```


If preferred, you can also wrap the agent executor in a `RunnableWithMessageHistory` class to internally manage history messages. First, we need to slightly modify the prompt to take a separate input variable so that the wrapper can parse which input value to store as history:

```
# Adapted from https://smith.langchain.com/hub/hwchase17/openai-tools-agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. You may not need to use tools for every query - the user may just want to chat!",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_openai_tools_agent(chat, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

```


Then, because our agent executor has multiple outputs, we also have to set the `output_messages_key` property when initializing the wrapper:

```
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

demo_ephemeral_chat_history_for_chain = ChatMessageHistory()

conversational_agent_executor = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: demo_ephemeral_chat_history_for_chain,
    input_messages_key="input",
    output_messages_key="output",
    history_messages_key="chat_history",
)

```


```
conversational_agent_executor.invoke(
    {
        "input": "I'm Nemo!",
    },
    {"configurable": {"session_id": "unused"}},
)

```


```


> Entering new AgentExecutor chain...
Hi Nemo! It's great to meet you. How can I assist you today?

> Finished chain.

```


```
{'input': "I'm Nemo!",
 'chat_history': [],
 'output': "Hi Nemo! It's great to meet you. How can I assist you today?"}

```


```
conversational_agent_executor.invoke(
    {
        "input": "What is my name?",
    },
    {"configurable": {"session_id": "unused"}},
)

```


```


> Entering new AgentExecutor chain...
Your name is Nemo! How can I assist you today, Nemo?

> Finished chain.

```


```
{'input': 'What is my name?',
 'chat_history': [HumanMessage(content="I'm Nemo!"),
  AIMessage(content="Hi Nemo! It's great to meet you. How can I assist you today?")],
 'output': 'Your name is Nemo! How can I assist you today, Nemo?'}

```


Further reading[​](#further-reading "Direct link to Further reading")
---------------------------------------------------------------------

Other types agents can also support conversational responses too - for more, check out the [agents section](https://python.langchain.com/docs/modules/agents/).

For more on tool usage, you can also check out [this use case section](https://python.langchain.com/docs/use_cases/tool_use/).