# Tool calling agent | 🦜️🔗 LangChain
[Tool calling](https://python.langchain.com/docs/modules/model_io/chat/function_calling/) allows a model to detect when one or more tools should be called and respond with the inputs that should be passed to those tools. In an API call, you can describe tools and have the model intelligently choose to output a structured object like JSON containing arguments to call these tools. The goal of tools APIs is to more reliably return valid and useful tool calls than what can be done using a generic text completion or chat API.

We can take advantage of this structured output, combined with the fact that you can bind multiple tools to a [tool calling chat model](https://python.langchain.com/docs/integrations/chat/) and allow the model to choose which one to call, to create an agent that repeatedly calls tools and receives results until a query is resolved.

This is a more generalized version of the [OpenAI tools agent](https://python.langchain.com/docs/modules/agents/agent_types/openai_tools/), which was designed for OpenAI’s specific style of tool calling. It uses LangChain’s ToolCall interface to support a wider range of provider implementations, such as [Anthropic](https://python.langchain.com/docs/integrations/chat/anthropic/), [Google Gemini](https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/), and [Mistral](https://python.langchain.com/docs/integrations/chat/mistralai/) in addition to [OpenAI](https://python.langchain.com/docs/integrations/chat/openai/).

Setup[​](#setup "Direct link to Setup")
---------------------------------------

Any models that support tool calling can be used in this agent. You can see which models support tool calling [here](https://python.langchain.com/docs/integrations/chat/)

This demo uses [Tavily](https://app.tavily.com/), but you can also swap in any other [built-in tool](https://python.langchain.com/docs/integrations/tools/) or add [custom tools](https://python.langchain.com/docs/modules/tools/custom_tools/). You’ll need to sign up for an API key and set it as `process.env.TAVILY_API_KEY`.

*   OpenAI
*   Anthropic
*   Google
*   FireworksAI
*   MistralAI
*   TogetherAI

##### Install dependencies

```
pip install -qU langchain-openai

```


##### Set environment variables

```
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

```


```
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

```


We will first create a tool that can search the web:

```
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

tools = [TavilySearchResults(max_results=1)]

```


Create Agent[​](#create-agent "Direct link to Create Agent")
------------------------------------------------------------

Next, let’s initialize our tool calling agent:

```
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Make sure to use the tavily_search_results_json tool for information.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Construct the Tools agent
agent = create_tool_calling_agent(llm, tools, prompt)

```


Run Agent[​](#run-agent "Direct link to Run Agent")
---------------------------------------------------

Now, let’s initialize the executor that will run our agent and invoke it!

```
# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "what is LangChain?"})

```


```


> Entering new AgentExecutor chain...

Invoking: `tavily_search_results_json` with `{'query': 'LangChain'}`
responded: [{'id': 'toolu_01QxrrT9srzkYCNyEZMDhGeg', 'input': {'query': 'LangChain'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]

[{'url': 'https://github.com/langchain-ai/langchain', 'content': 'About\n⚡ Building applications with LLMs through composability ⚡\nResources\nLicense\nCode of conduct\nSecurity policy\nStars\nWatchers\nForks\nReleases\n291\nPackages\n0\nUsed by 39k\nContributors\n1,848\nLanguages\nFooter\nFooter navigation Latest commit\nGit stats\nFiles\nREADME.md\n🦜️🔗 LangChain\n⚡ Building applications with LLMs through composability ⚡\nLooking for the JS/TS library? ⚡ Building applications with LLMs through composability ⚡\nLicense\nlangchain-ai/langchain\nName already in use\nUse Git or checkout with SVN using the web URL.\n 📖 Documentation\nPlease see here for full documentation, which includes:\n💁 Contributing\nAs an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.\n What can you build with LangChain?\n❓ Retrieval augmented generation\n💬 Analyzing structured data\n🤖 Chatbots\nAnd much more!'}]LangChain is an open-source Python library that helps developers build applications with large language models (LLMs) through composability. Some key features of LangChain include:

- Retrieval augmented generation - Allowing LLMs to retrieve and utilize external data sources when generating outputs.

- Analyzing structured data - Tools for working with structured data like databases, APIs, PDFs, etc. and allowing LLMs to reason over this data.

- Building chatbots and agents - Frameworks for building conversational AI applications.

- Composability - LangChain allows you to chain together different LLM capabilities and data sources in a modular and reusable way.

The library aims to make it easier to build real-world applications that leverage the power of large language models in a scalable and robust way. It provides abstractions and primitives for working with LLMs from different providers like OpenAI, Anthropic, Cohere, etc. LangChain is open-source and has an active community contributing new features and improvements.

> Finished chain.

```


```
/Users/bagatur/langchain/libs/partners/anthropic/langchain_anthropic/chat_models.py:347: UserWarning: stream: Tool use is not yet supported in streaming mode.
  warnings.warn("stream: Tool use is not yet supported in streaming mode.")
/Users/bagatur/langchain/libs/partners/anthropic/langchain_anthropic/chat_models.py:347: UserWarning: stream: Tool use is not yet supported in streaming mode.
  warnings.warn("stream: Tool use is not yet supported in streaming mode.")

```


```
{'input': 'what is LangChain?',
 'output': 'LangChain is an open-source Python library that helps developers build applications with large language models (LLMs) through composability. Some key features of LangChain include:\n\n- Retrieval augmented generation - Allowing LLMs to retrieve and utilize external data sources when generating outputs.\n\n- Analyzing structured data - Tools for working with structured data like databases, APIs, PDFs, etc. and allowing LLMs to reason over this data.\n\n- Building chatbots and agents - Frameworks for building conversational AI applications.\n\n- Composability - LangChain allows you to chain together different LLM capabilities and data sources in a modular and reusable way.\n\nThe library aims to make it easier to build real-world applications that leverage the power of large language models in a scalable and robust way. It provides abstractions and primitives for working with LLMs from different providers like OpenAI, Anthropic, Cohere, etc. LangChain is open-source and has an active community contributing new features and improvements.'}

```


Using with chat history[​](#using-with-chat-history "Direct link to Using with chat history")
---------------------------------------------------------------------------------------------

This type of agent can optionally take chat messages representing previous conversation turns. It can use that previous history to respond conversationally. For more details, see [this section of the agent quickstart](https://python.langchain.com/docs/modules/agents/quick_start/#adding-in-memory).

```
from langchain_core.messages import AIMessage, HumanMessage

agent_executor.invoke(
    {
        "input": "what's my name? Don't use tools to look this up unless you NEED to",
        "chat_history": [
            HumanMessage(content="hi! my name is bob"),
            AIMessage(content="Hello Bob! How can I assist you today?"),
        ],
    }
)

```


```


> Entering new AgentExecutor chain...
Based on what you told me, your name is Bob. I don't need to use any tools to look that up since you directly provided your name.

> Finished chain.

```


```
/Users/bagatur/langchain/libs/partners/anthropic/langchain_anthropic/chat_models.py:347: UserWarning: stream: Tool use is not yet supported in streaming mode.
  warnings.warn("stream: Tool use is not yet supported in streaming mode.")

```


```
{'input': "what's my name? Don't use tools to look this up unless you NEED to",
 'chat_history': [HumanMessage(content='hi! my name is bob'),
  AIMessage(content='Hello Bob! How can I assist you today?')],
 'output': "Based on what you told me, your name is Bob. I don't need to use any tools to look that up since you directly provided your name."}

```
