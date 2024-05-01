# Toolkits | 🦜️🔗 LangChain
*   [](https://python.langchain.com/)
*   [Composition](https://python.langchain.com/docs/modules/composition/)
*   [Tools](https://python.langchain.com/docs/modules/tools/)
*   Toolkits

Toolkits are collections of tools that are designed to be used together for specific tasks. They have convenient loading methods. For a complete list of available ready-made toolkits, visit [Integrations](https://python.langchain.com/docs/integrations/toolkits/).

All Toolkits expose a `get_tools` method which returns a list of tools. You can therefore do:

```
# Initialize a toolkit
toolkit = ExampleTookit(...)

# Get list of tools
tools = toolkit.get_tools()

# Create agent
agent = create_agent_method(llm, tools, prompt)

```


* * *

#### Help us out by providing feedback on this documentation page:

[

Previous

Tools

](https://python.langchain.com/docs/modules/tools/)[

Next

Tools

](https://python.langchain.com/docs/modules/tools/)