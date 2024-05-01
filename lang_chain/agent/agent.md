# Agents | 🦜️🔗 LangChain
The core idea of agents is to use a language model to choose a sequence of actions to take. In chains, a sequence of actions is hardcoded (in code). In agents, a language model is used as a reasoning engine to determine which actions to take and in which order.

[Quickstart](https://python.langchain.com/docs/modules/agents/quick_start/)[​](#quickstart "Direct link to quickstart")
-----------------------------------------------------------------------------------------------------------------------

For a quick start to working with agents, please check out [this getting started guide](https://python.langchain.com/docs/modules/agents/quick_start/). This covers basics like initializing an agent, creating tools, and adding memory.

[Concepts](https://python.langchain.com/docs/modules/agents/concepts/)[​](#concepts "Direct link to concepts")
--------------------------------------------------------------------------------------------------------------

There are several key concepts to understand when building agents: Agents, AgentExecutor, Tools, Toolkits. For an in depth explanation, please check out [this conceptual guide](https://python.langchain.com/docs/modules/agents/concepts/)

[Agent Types](https://python.langchain.com/docs/modules/agents/agent_types/)[​](#agent-types "Direct link to agent-types")
--------------------------------------------------------------------------------------------------------------------------

There are many different types of agents to use. For a overview of the different types and when to use them, please check out [this section](https://python.langchain.com/docs/modules/agents/agent_types/).

Agents are only as good as the tools they have. For a comprehensive guide on tools, please see [this section](https://python.langchain.com/docs/modules/tools/).

How To Guides[​](#how-to-guides "Direct link to How To Guides")
---------------------------------------------------------------

Agents have a lot of related functionality! Check out comprehensive guides including:

*   [Building a custom agent](https://python.langchain.com/docs/modules/agents/how_to/custom_agent/)
*   [Streaming (of both intermediate steps and tokens](https://python.langchain.com/docs/modules/agents/how_to/streaming/)
*   [Building an agent that returns structured output](https://python.langchain.com/docs/modules/agents/how_to/agent_structured/)
*   Lots functionality around using AgentExecutor, including: [using it as an iterator](https://python.langchain.com/docs/modules/agents/how_to/agent_iter/), [handle parsing errors](https://python.langchain.com/docs/modules/agents/how_to/handle_parsing_errors/), [returning intermediate steps](https://python.langchain.com/docs/modules/agents/how_to/intermediate_steps/), [capping the max number of iterations](https://python.langchain.com/docs/modules/agents/how_to/max_iterations/), and [timeouts for agents](https://python.langchain.com/docs/modules/agents/how_to/max_time_limit/)
