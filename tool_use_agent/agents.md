# Repeated tool use with agents | 🦜️🔗 LangChain
Chains are great when we know the specific sequence of tool usage needed for any user input. But for certain use cases, how many times we use tools depends on the input. In these cases, we want to let the model itself decide how many times to use tools and in what order. [Agents](https://python.langchain.com/docs/modules/agents/) let us do just this.

LangChain comes with a number of built-in agents that are optimized for different use cases. Read about all the [agent types here](https://python.langchain.com/docs/modules/agents/agent_types/).

We’ll use the [tool calling agent](https://python.langchain.com/docs/modules/agents/agent_types/tool_calling/), which is generally the most reliable kind and the recommended one for most use cases. “Tool calling” in this case refers to a specific type of model API that allows for explicitly passing tool definitions to models and getting explicit tool invocations out. For more on tool calling models see [this guide](https://python.langchain.com/docs/modules/model_io/chat/function_calling/).

First, we need to create some tool to call. For this example, we will create custom tools from functions. For more information on creating custom tools, please see [this guide](https://python.langchain.com/docs/modules/tools/).

```
================================ System Message ================================

You are a helpful assistant

============================= Messages Placeholder =============================

{chat_history}

================================ Human Message =================================

{input}

============================= Messages Placeholder =============================

{agent_scratchpad}

```


We’ll need to use a model with tool calling capabilities. You can see which models support tool calling [here](https://python.langchain.com/docs/integrations/chat/).

```


> Entering new AgentExecutor chain...

Invoking: `exponentiate` with `{'base': 3, 'exponent': 5}`
responded: [{'text': "Okay, let's break this down step-by-step:", 'type': 'text'}, {'id': 'toolu_01CjdiDhDmMtaT1F4R7hSV5D', 'input': {'base': 3, 'exponent': 5}, 'name': 'exponentiate', 'type': 'tool_use'}]

243
Invoking: `add` with `{'first_int': 12, 'second_int': 3}`
responded: [{'text': '3 to the 5th power is 243.', 'type': 'text'}, {'id': 'toolu_01EKqn4E5w3Zj7bQ8s8xmi4R', 'input': {'first_int': 12, 'second_int': 3}, 'name': 'add', 'type': 'tool_use'}]

15
Invoking: `multiply` with `{'first_int': 243, 'second_int': 15}`
responded: [{'text': '12 + 3 = 15', 'type': 'text'}, {'id': 'toolu_017VZJgZBYbwMo2KGD6o6hsQ', 'input': {'first_int': 243, 'second_int': 15}, 'name': 'multiply', 'type': 'tool_use'}]

3645
Invoking: `multiply` with `{'first_int': 3645, 'second_int': 3645}`
responded: [{'text': '243 * 15 = 3645', 'type': 'text'}, {'id': 'toolu_01RtFCcQgbVGya3NVDgTYKTa', 'input': {'first_int': 3645, 'second_int': 3645}, 'name': 'multiply', 'type': 'tool_use'}]

13286025So 3645 squared is 13,286,025.

Therefore, the final result of taking 3 to the 5th power (243), multiplying by 12 + 3 (15), and then squaring the whole result is 13,286,025.

> Finished chain.

```


```
{'input': 'Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result',
 'output': 'So 3645 squared is 13,286,025.\n\nTherefore, the final result of taking 3 to the 5th power (243), multiplying by 12 + 3 (15), and then squaring the whole result is 13,286,025.'}

```
