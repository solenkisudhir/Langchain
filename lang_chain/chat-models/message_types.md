# Message types | 🦜️🔗 LangChain
ChatModels take a list of messages as input and return a message. There are a few different types of messages. All messages have a `role` and a `content` property. The `role` describes WHO is saying the message. LangChain has different message classes for different roles. The `content` property describes the content of the message. This can be a few different things:

*   A string (most models deal this type of content)
*   A List of dictionaries (this is used for multi-modal input, where the dictionary contains information about that input type and that input location)

In addition, messages have an `additional_kwargs` property. This is where additional information about messages can be passed. This is largely used for input parameters that are _provider specific_ and not general. The best known example of this is `function_call` from OpenAI.

### HumanMessage[​](#humanmessage "Direct link to HumanMessage")

This represents a message from the user. Generally consists only of content.

### AIMessage[​](#aimessage "Direct link to AIMessage")

This represents a message from the model. This may have `additional_kwargs` in it - for example `tool_calls` if using OpenAI tool calling.

### SystemMessage[​](#systemmessage "Direct link to SystemMessage")

This represents a system message, which tells the model how to behave. This generally only consists of content. Not every model supports this.

### FunctionMessage[​](#functionmessage "Direct link to FunctionMessage")

This represents the result of a function call. In addition to `role` and `content`, this message has a `name` parameter which conveys the name of the function that was called to produce this result.

### ToolMessage[​](#toolmessage "Direct link to ToolMessage")

This represents the result of a tool call. This is distinct from a FunctionMessage in order to match OpenAI's `function` and `tool` message types. In addition to `role` and `content`, this message has a `tool_call_id` parameter which conveys the id of the call to the tool that was called to produce this result.