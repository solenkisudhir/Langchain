# Self-ask with search | 🦜️🔗 LangChain
This walkthrough showcases the self-ask with search agent.

```
from langchain import hub
from langchain.agents import AgentExecutor, create_self_ask_with_search_agent
from langchain_community.llms import Fireworks
from langchain_community.tools.tavily_search import TavilyAnswer

```


We will initialize the tools we want to use. This is a good tool because it gives us **answers** (not documents)

For this agent, only one tool can be used and it needs to be named “Intermediate Answer”

```
tools = [TavilyAnswer(max_results=1, name="Intermediate Answer")]

```


Create Agent[​](#create-agent "Direct link to Create Agent")
------------------------------------------------------------

```
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/self-ask-with-search")

```


```
# Choose the LLM that will drive the agent
llm = Fireworks()

# Construct the Self Ask With Search Agent
agent = create_self_ask_with_search_agent(llm, tools, prompt)

```


Run Agent[​](#run-agent "Direct link to Run Agent")
---------------------------------------------------

```
# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

```


```
agent_executor.invoke(
    {"input": "What is the hometown of the reigning men's U.S. Open champion?"}
)

```


```


> Entering new AgentExecutor chain...
 Yes.
Follow up: Who is the reigning men's U.S. Open champion?The reigning men's U.S. Open champion is Novak Djokovic. He won his 24th Grand Slam singles title by defeating Daniil Medvedev in the final of the 2023 U.S. Open.
So the final answer is: Novak Djokovic.

> Finished chain.

```


```
{'input': "What is the hometown of the reigning men's U.S. Open champion?",
 'output': 'Novak Djokovic.'}

```


* * *

#### Help us out by providing feedback on this documentation page:
