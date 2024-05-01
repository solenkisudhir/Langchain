# Tool use and agents | 🦜️🔗 LangChain
An exciting use case for LLMs is building natural language interfaces for other “tools”, whether those are APIs, functions, databases, etc. LangChain is great for building such interfaces because it has:

*   Good model output parsing, which makes it easy to extract JSON, XML, OpenAI function-calls, etc. from model outputs.
*   A large collection of built-in [Tools](https://python.langchain.com/docs/integrations/tools/).
*   Provides a lot of flexibility in how you call these tools.

There are two main ways to use tools: [chains](https://python.langchain.com/docs/modules/chains/) and [agents](https://python.langchain.com/docs/modules/agents/).

Chains lets you create a pre-defined sequence of tool usage(s).

![chain](https://python.langchain.com/assets/images/tool_chain-3571e7fbc481d648aff93a2630f812ab.svg)

Agents let the model use tools in a loop, so that it can decide how many times to use tools.

![agent](https://python.langchain.com/assets/images/tool_agent-d25fafc271da3ee950ac1fba59cdf490.svg)

To get started with both approaches, head to the [Quickstart](https://python.langchain.com/docs/use_cases/tool_use/quickstart/) page.