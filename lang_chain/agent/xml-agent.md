# XML Agent | 🦜️🔗 LangChain
Some language models (like Anthropic’s Claude) are particularly good at reasoning/writing XML. This goes over how to use an agent that uses XML when prompting.

tip

*   Use with regular LLMs, not with chat models.
*   Use only with unstructured tools; i.e., tools that accept a single string input.
*   See [AgentTypes](https://python.langchain.com/docs/modules/agents/agent_types/) documentation for more agent types.

```
from langchain import hub
from langchain.agents import AgentExecutor, create_xml_agent
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults

```


We will initialize the tools we want to use

```
tools = [TavilySearchResults(max_results=1)]

```


Create Agent[​](#create-agent "Direct link to Create Agent")
------------------------------------------------------------

Below we will use LangChain’s built-in [create\_xml\_agent](https://api.python.langchain.com/en/latest/agents/langchain.agents.xml.base.create_xml_agent.html) constructor.

```
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/xml-agent-convo")

```


```
# Choose the LLM that will drive the agent
llm = ChatAnthropic(model="claude-2.1")

# Construct the XML agent
agent = create_xml_agent(llm, tools, prompt)

```


Run Agent[​](#run-agent "Direct link to Run Agent")
---------------------------------------------------

```
# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

```


```
agent_executor.invoke({"input": "what is LangChain?"})

```


```


> Entering new AgentExecutor chain...
 <tool>tavily_search_results_json</tool><tool_input>what is LangChain?[{'url': 'https://aws.amazon.com/what-is/langchain/', 'content': 'What Is LangChain? What is LangChain?  How does LangChain work?  Why is LangChain important?  that LangChain provides to reduce development time.LangChain is an open source framework for building applications based on large language models (LLMs). LLMs are large deep-learning models pre-trained on large amounts of data that can generate responses to user queries—for example, answering questions or creating images from text-based prompts.'}] <final_answer>LangChain is an open source framework for building applications based on large language models (LLMs). It allows developers to leverage the power of LLMs to create applications that can generate responses to user queries, such as answering questions or creating images from text prompts. Key benefits of LangChain are reducing development time and effort compared to building custom LLMs from scratch.</final_answer>

> Finished chain.

```


```
{'input': 'what is LangChain?',
 'output': 'LangChain is an open source framework for building applications based on large language models (LLMs). It allows developers to leverage the power of LLMs to create applications that can generate responses to user queries, such as answering questions or creating images from text prompts. Key benefits of LangChain are reducing development time and effort compared to building custom LLMs from scratch.'}

```


Using with chat history[​](#using-with-chat-history "Direct link to Using with chat history")
---------------------------------------------------------------------------------------------

```
from langchain_core.messages import AIMessage, HumanMessage

agent_executor.invoke(
    {
        "input": "what's my name? Only use a tool if needed, otherwise respond with Final Answer",
        # Notice that chat_history is a string, since this prompt is aimed at LLMs, not chat models
        "chat_history": "Human: Hi! My name is Bob\nAI: Hello Bob! Nice to meet you",
    }
)

```


```


> Entering new AgentExecutor chain...
 <final_answer>Your name is Bob.</final_answer>

Since you already told me your name is Bob, I do not need to use any tools to answer the question "what's my name?". I can provide the final answer directly that your name is Bob.

> Finished chain.

```


```
{'input': "what's my name? Only use a tool if needed, otherwise respond with Final Answer",
 'chat_history': 'Human: Hi! My name is Bob\nAI: Hello Bob! Nice to meet you',
 'output': 'Your name is Bob.'}

```


Custom XML Agents
-----------------

**Note:** For greater customizability, we recommend checking out [LangGraph](https://python.langchain.com/docs/langgraph/).

Here we provide an example of a custom XML Agent implementation, to give a sense for what `create_xml_agent` is doing under the hood.

```
from langchain.agents.output_parsers import XMLAgentOutputParser

```


```
# Logic for going from intermediate steps to a string to pass into model
# This is pretty tied to the prompt
def convert_intermediate_steps(intermediate_steps):
    log = ""
    for action, observation in intermediate_steps:
        log += (
            f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
            f"</tool_input><observation>{observation}</observation>"
        )
    return log


# Logic for converting tools to string to go in prompt
def convert_tools(tools):
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

```


Building an agent from a runnable usually involves a few things:

1.  Data processing for the intermediate steps. These need to be represented in a way that the language model can recognize them. This should be pretty tightly coupled to the instructions in the prompt
    
2.  The prompt itself
    
3.  The model, complete with stop tokens if needed
    
4.  The output parser - should be in sync with how the prompt specifies things to be formatted.
    

```
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: convert_intermediate_steps(
            x["intermediate_steps"]
        ),
    }
    | prompt.partial(tools=convert_tools(tools))
    | llm.bind(stop=["</tool_input>", "</final_answer>"])
    | XMLAgentOutputParser()
)

```


```
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

```


```
agent_executor.invoke({"input": "what is LangChain?"})

```


```


> Entering new AgentExecutor chain...
<tool>tavily_search_results_json</tool>
<tool_input>what is LangChain[{'url': 'https://www.techtarget.com/searchEnterpriseAI/definition/LangChain', 'content': "Everything you need to know\nWhat are the features of LangChain?\nLangChain is made up of the following modules that ensure the multiple components needed to make an effective NLP app can run smoothly:\nWhat are the integrations of LangChain?\nLangChain typically builds applications using integrations with LLM providers and external sources where data can be found and stored. What is synthetic data?\nExamples and use cases for LangChain\nThe LLM-based applications LangChain is capable of building can be applied to multiple advanced use cases within various industries and vertical markets, such as the following:\nReaping the benefits of NLP is a key of why LangChain is important. As the airline giant moves more of its data workloads to the cloud, tools from Intel's Granulate are making platforms such as ...\nThe vendor's new platform, now in beta testing, combines its existing lakehouse with AI to better enable users to manage and ...\n The following steps are required to use this:\nIn this scenario, the language model would be expected to take the two input variables -- the adjective and the content -- and produce a fascinating fact about zebras as its output.\n The goal of LangChain is to link powerful LLMs, such as OpenAI's GPT-3.5 and GPT-4, to an array of external data sources to create and reap the benefits of natural language processing (NLP) applications.\n"}]<final_answer>
LangChain is a platform developed by Anthropic that enables users to build NLP applications by linking large language models like GPT-3.5 and GPT-4 to external data sources. It provides modules for managing and integrating different components needed for NLP apps.

Some key capabilities and features of LangChain:

- Allows linking LLMs to external data sources to create customized NLP apps
- Provides modules to manage integration of LLMs, data sources, storage etc. 
- Enables building conversational AI apps, summarization, search, and other NLP capabilities
- Helps users reap benefits of NLP and LLMs for use cases across industries

So in summary, it is a platform to build and deploy advanced NLP models by leveraging capabilities of large language models in a more customizable and scalable way.



> Finished chain.

```


```
{'input': 'what is LangChain?',
 'output': '\nLangChain is a platform developed by Anthropic that enables users to build NLP applications by linking large language models like GPT-3.5 and GPT-4 to external data sources. It provides modules for managing and integrating different components needed for NLP apps.\n\nSome key capabilities and features of LangChain:\n\n- Allows linking LLMs to external data sources to create customized NLP apps\n- Provides modules to manage integration of LLMs, data sources, storage etc. \n- Enables building conversational AI apps, summarization, search, and other NLP capabilities\n- Helps users reap benefits of NLP and LLMs for use cases across industries\n\nSo in summary, it is a platform to build and deploy advanced NLP models by leveraging capabilities of large language models in a more customizable and scalable way.\n\n'}

```
