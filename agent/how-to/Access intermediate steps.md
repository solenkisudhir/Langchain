# Access intermediate steps | 🦜️🔗 LangChain
In order to get more visibility into what an agent is doing, we can also return intermediate steps. This comes in the form of an extra key in the return value, which is a list of (action, observation) tuples.

```
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
tool = WikipediaQueryRun(api_wrapper=api_wrapper)
tools = [tool]

# Get the prompt to use - you can modify this!
# If you want to see the prompt in full, you can at: https://smith.langchain.com/hub/hwchase17/openai-functions-agent
prompt = hub.pull("hwchase17/openai-functions-agent")

llm = ChatOpenAI(temperature=0)

agent = create_openai_functions_agent(llm, tools, prompt)

```


Initialize the AgentExecutor with `return_intermediate_steps=True`:

```
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
)

```


```
response = agent_executor.invoke({"input": "What is Leo DiCaprio's middle name?"})

```


```


> Entering new AgentExecutor chain...

Invoking: `Wikipedia` with `Leo DiCaprio`


Page: Leonardo DiCaprio
Summary: Leonardo Wilhelm DiCaprio (; Italian: [diˈkaːprjo]; born November 1Leonardo DiCaprio's middle name is Wilhelm.

> Finished chain.

```


```
# The actual return type is a NamedTuple for the agent action, and then an observation
print(response["intermediate_steps"])

```


```
[(AgentActionMessageLog(tool='Wikipedia', tool_input='Leo DiCaprio', log='\nInvoking: `Wikipedia` with `Leo DiCaprio`\n\n\n', message_log=[AIMessage(content='', additional_kwargs={'function_call': {'name': 'Wikipedia', 'arguments': '{\n  "__arg1": "Leo DiCaprio"\n}'}})]), 'Page: Leonardo DiCaprio\nSummary: Leonardo Wilhelm DiCaprio (; Italian: [diˈkaːprjo]; born November 1')]

```


* * *

#### Help us out by providing feedback on this documentation page: