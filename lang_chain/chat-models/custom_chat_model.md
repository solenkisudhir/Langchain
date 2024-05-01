# Custom Chat Model | 🦜️🔗 LangChain
In this guide, we’ll learn how to create a custom chat model using LangChain abstractions.

Wrapping your LLM with the standard `BaseChatModel` interface allow you to use your LLM in existing LangChain programs with minimal code modifications!

As an bonus, your LLM will automatically become a LangChain `Runnable` and will benefit from some optimizations out of the box (e.g., batch via a threadpool), async support, the `astream_events` API, etc.

Inputs and outputs[​](#inputs-and-outputs "Direct link to Inputs and outputs")
------------------------------------------------------------------------------

First, we need to talk about **messages** which are the inputs and outputs of chat models.

### Messages[​](#messages "Direct link to Messages")

Chat models take messages as inputs and return a message as output.

LangChain has a few built-in message types:



* Message Type: SystemMessage
  * Description: Used for priming AI behavior, usually passed in as the first of a sequence of input messages.
* Message Type: HumanMessage
  * Description: Represents a message from a person interacting with the chat model.
* Message Type: AIMessage
  * Description: Represents a message from the chat model. This can be either text or a request to invoke a tool.
* Message Type: FunctionMessage / ToolMessage
  * Description: Message for passing the results of tool invocation back to the model.
* Message Type: AIMessageChunk / HumanMessageChunk / …
  * Description: Chunk variant of each type of message.


note

`ToolMessage` and `FunctionMessage` closely follow OpenAIs `function` and `tool` roles.

This is a rapidly developing field and as more models add function calling capabilities, expect that there will be additions to this schema.

```
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

```


### Streaming Variant[​](#streaming-variant "Direct link to Streaming Variant")

All the chat messages have a streaming variant that contains `Chunk` in the name.

```
from langchain_core.messages import (
    AIMessageChunk,
    FunctionMessageChunk,
    HumanMessageChunk,
    SystemMessageChunk,
    ToolMessageChunk,
)

```


These chunks are used when streaming output from chat models, and they all define an additive property!

```
AIMessageChunk(content="Hello") + AIMessageChunk(content=" World!")

```


```
AIMessageChunk(content='Hello World!')

```


Base Chat Model[​](#base-chat-model "Direct link to Base Chat Model")
---------------------------------------------------------------------

Let’s implement a chat model that echoes back the first `n` characetrs of the last message in the prompt!

To do so, we will inherit from `BaseChatModel` and we’ll need to implement the following:



* Method/Property: _generate
  * Description: Use to generate a chat result from a prompt
  * Required/Optional: Required
* Method/Property: _llm_type (property)
  * Description: Used to uniquely identify the type of the model. Used for logging.
  * Required/Optional: Required
* Method/Property: _identifying_params (property)
  * Description: Represent model parameterization for tracing purposes.
  * Required/Optional: Optional
* Method/Property: _stream
  * Description: Use to implement streaming.
  * Required/Optional: Optional
* Method/Property: _agenerate
  * Description: Use to implement a native async method.
  * Required/Optional: Optional
* Method/Property: _astream
  * Description: Use to implement async version of _stream.
  * Required/Optional: Optional


tip

The `_astream` implementation uses `run_in_executor` to launch the sync `_stream` in a separate thread if `_stream` is implemented, otherwise it fallsback to use `_agenerate`.

You can use this trick if you want to reuse the `_stream` implementation, but if you’re able to implement code that’s natively async that’s a better solution since that code will run with less overhead.

### Implementation[​](#implementation "Direct link to Implementation")

```
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import run_in_executor


class CustomChatModelAdvanced(BaseChatModel):
    """A custom chat model that echoes the first `n` characters of the input.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.

    Example:

        .. code-block:: python

            model = CustomChatModel(n=2)
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                 [HumanMessage(content="world")]])
    """

    model_name: str
    """The name of the model"""
    n: int
    """The number of characters from the last message of the prompt to be echoed."""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        # Replace this with actual logic to generate a response from a list
        # of messages.
        last_message = messages[-1]
        tokens = last_message.content[: self.n]
        message = AIMessage(
            content=tokens,
            additional_kwargs={},  # Used to add additional payload (e.g., function calling request)
            response_metadata={  # Use for response metadata
                "time_in_seconds": 3,
            },
        )
        ##

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model.

        This method should be implemented if the model can generate output
        in a streaming fashion. If the model does not support streaming,
        do not implement it. In that case streaming requests will be automatically
        handled by the _generate method.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        last_message = messages[-1]
        tokens = last_message.content[: self.n]

        for token in tokens:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))

            if run_manager:
                # This is optional in newer versions of LangChain
                # The on_llm_new_token will be called automatically
                run_manager.on_llm_new_token(token, chunk=chunk)

            yield chunk

        # Let's add some other information (e.g., response metadata)
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata={"time_in_sec": 3})
        )
        if run_manager:
            # This is optional in newer versions of LangChain
            # The on_llm_new_token will be called automatically
            run_manager.on_llm_new_token(token, chunk=chunk)
        yield chunk

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }

```


#### API Reference:

*   [AsyncCallbackManagerForLLMRun](https://api.python.langchain.com/en/latest/callbacks/langchain_core.callbacks.manager.AsyncCallbackManagerForLLMRun.html)
*   [CallbackManagerForLLMRun](https://api.python.langchain.com/en/latest/callbacks/langchain_core.callbacks.manager.CallbackManagerForLLMRun.html)
*   [BaseChatModel](https://api.python.langchain.com/en/latest/language_models/langchain_core.language_models.chat_models.BaseChatModel.html)
*   [SimpleChatModel](https://api.python.langchain.com/en/latest/language_models/langchain_core.language_models.chat_models.SimpleChatModel.html)
*   [AIMessageChunk](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.ai.AIMessageChunk.html)
*   [BaseMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.base.BaseMessage.html)
*   [HumanMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.human.HumanMessage.html)
*   [ChatGeneration](https://api.python.langchain.com/en/latest/outputs/langchain_core.outputs.chat_generation.ChatGeneration.html)
*   [ChatGenerationChunk](https://api.python.langchain.com/en/latest/outputs/langchain_core.outputs.chat_generation.ChatGenerationChunk.html)
*   [ChatResult](https://api.python.langchain.com/en/latest/outputs/langchain_core.outputs.chat_result.ChatResult.html)
*   [run\_in\_executor](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.config.run_in_executor.html)

### Let’s test it 🧪[​](#lets-test-it "Direct link to Let’s test it 🧪")

The chat model will implement the standard `Runnable` interface of LangChain which many of the LangChain abstractions support!

```
model = CustomChatModelAdvanced(n=3, model_name="my_custom_model")

```


```
model.invoke(
    [
        HumanMessage(content="hello!"),
        AIMessage(content="Hi there human!"),
        HumanMessage(content="Meow!"),
    ]
)

```


```
AIMessage(content='Meo', response_metadata={'time_in_seconds': 3}, id='run-ddb42bd6-4fdd-4bd2-8be5-e11b67d3ac29-0')

```


```
AIMessage(content='hel', response_metadata={'time_in_seconds': 3}, id='run-4d3cc912-44aa-454b-977b-ca02be06c12e-0')

```


```
model.batch(["hello", "goodbye"])

```


```
[AIMessage(content='hel', response_metadata={'time_in_seconds': 3}, id='run-9620e228-1912-4582-8aa1-176813afec49-0'),
 AIMessage(content='goo', response_metadata={'time_in_seconds': 3}, id='run-1ce8cdf8-6f75-448e-82f7-1bb4a121df93-0')]

```


```
for chunk in model.stream("cat"):
    print(chunk.content, end="|")

```


Please see the implementation of `_astream` in the model! If you do not implement it, then no output will stream.!

```
async for chunk in model.astream("cat"):
    print(chunk.content, end="|")

```


Let’s try to use the astream events API which will also help double check that all the callbacks were implemented!

```
async for event in model.astream_events("cat", version="v1"):
    print(event)

```


```
{'event': 'on_chat_model_start', 'run_id': '125a2a16-b9cd-40de-aa08-8aa9180b07d0', 'name': 'CustomChatModelAdvanced', 'tags': [], 'metadata': {}, 'data': {'input': 'cat'}}
{'event': 'on_chat_model_stream', 'run_id': '125a2a16-b9cd-40de-aa08-8aa9180b07d0', 'tags': [], 'metadata': {}, 'name': 'CustomChatModelAdvanced', 'data': {'chunk': AIMessageChunk(content='c', id='run-125a2a16-b9cd-40de-aa08-8aa9180b07d0')}}
{'event': 'on_chat_model_stream', 'run_id': '125a2a16-b9cd-40de-aa08-8aa9180b07d0', 'tags': [], 'metadata': {}, 'name': 'CustomChatModelAdvanced', 'data': {'chunk': AIMessageChunk(content='a', id='run-125a2a16-b9cd-40de-aa08-8aa9180b07d0')}}
{'event': 'on_chat_model_stream', 'run_id': '125a2a16-b9cd-40de-aa08-8aa9180b07d0', 'tags': [], 'metadata': {}, 'name': 'CustomChatModelAdvanced', 'data': {'chunk': AIMessageChunk(content='t', id='run-125a2a16-b9cd-40de-aa08-8aa9180b07d0')}}
{'event': 'on_chat_model_stream', 'run_id': '125a2a16-b9cd-40de-aa08-8aa9180b07d0', 'tags': [], 'metadata': {}, 'name': 'CustomChatModelAdvanced', 'data': {'chunk': AIMessageChunk(content='', response_metadata={'time_in_sec': 3}, id='run-125a2a16-b9cd-40de-aa08-8aa9180b07d0')}}
{'event': 'on_chat_model_end', 'name': 'CustomChatModelAdvanced', 'run_id': '125a2a16-b9cd-40de-aa08-8aa9180b07d0', 'tags': [], 'metadata': {}, 'data': {'output': AIMessageChunk(content='cat', response_metadata={'time_in_sec': 3}, id='run-125a2a16-b9cd-40de-aa08-8aa9180b07d0')}}

```


```
/home/eugene/src/langchain/libs/core/langchain_core/_api/beta_decorator.py:87: LangChainBetaWarning: This API is in beta and may change in the future.
  warn_beta(

```


Contributing[​](#contributing "Direct link to Contributing")
------------------------------------------------------------

We appreciate all chat model integration contributions.

Here’s a checklist to help make sure your contribution gets added to LangChain:

Documentation:

*   The model contains doc-strings for all initialization arguments, as these will be surfaced in the [APIReference](https://api.python.langchain.com/en/stable/langchain_api_reference.html).
*   The class doc-string for the model contains a link to the model API if the model is powered by a service.

Tests:

*   ☐ Add unit or integration tests to the overridden methods. Verify that `invoke`, `ainvoke`, `batch`, `stream` work if you’ve over-ridden the corresponding code.

Streaming (if you’re implementing it):

*   ☐ Implement the \_stream method to get streaming working

Stop Token Behavior:

*   ☐ Stop token should be respected
*   ☐ Stop token should be INCLUDED as part of the response

Secret API Keys:

*   ☐ If your model connects to an API it will likely accept API keys as part of its initialization. Use Pydantic’s `SecretStr` type for secrets, so they don’t get accidentally printed out when folks print the model.

Identifying Params:

*   ☐ Include a `model_name` in identifying params

Optimizations:

Consider providing native async support to reduce the overhead from the model!

*   ☐ Provided a native async of `_agenerate` (used by `ainvoke`)
*   ☐ Provided a native async of `_astream` (used by `astream`)