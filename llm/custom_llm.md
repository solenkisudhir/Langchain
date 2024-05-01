# Custom LLM | 🦜️🔗 LangChain
This notebook goes over how to create a custom LLM wrapper, in case you want to use your own LLM or a different wrapper than one that is supported in LangChain.

Wrapping your LLM with the standard `LLM` interface allow you to use your LLM in existing LangChain programs with minimal code modifications!

As an bonus, your LLM will automatically become a LangChain `Runnable` and will benefit from some optimizations out of the box, async support, the `astream_events` API, etc.

Implementation[​](#implementation "Direct link to Implementation")
------------------------------------------------------------------

There are only two required things that a custom LLM needs to implement:


|Method   |Description                                                                          |
|---------|-------------------------------------------------------------------------------------|
|_call    |Takes in a string and some optional stop words, and returns a string. Used by invoke.|
|_llm_type|A property that returns a string, used for logging purposes only.                    |


Optional implementations:



* Method: _identifying_params
  * Description: Used to help with identifying the model and printing the LLM; should return a dictionary. This is a @property.
* Method: _acall
  * Description: Provides an async native implementation of _call, used by ainvoke.
* Method: _stream
  * Description: Method to stream the output token by token.
* Method: _astream
  * Description: Provides an async native implementation of _stream; in newer LangChain versions, defaults to _stream.


Let’s implement a simple custom LLM that just returns the first n characters of the input.

```
from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk


class CustomLLM(LLM):
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

    n: int
    """The number of characters from the last message of the prompt to be echoed."""

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[: self.n]

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM on the given prompt.

        This method should be overridden by subclasses that support streaming.

        If not implemented, the default behavior of calls to stream will be to
        fallback to the non-streaming version of the model and return
        the output as a single chunk.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An iterator of GenerationChunks.
        """
        for char in prompt[: self.n]:
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "CustomChatModel",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"

```


### Let’s test it 🧪[​](#lets-test-it "Direct link to Let’s test it 🧪")

This LLM will implement the standard `Runnable` interface of LangChain which many of the LangChain abstractions support!

```
llm = CustomLLM(n=5)
print(llm)

```


```
CustomLLM
Params: {'model_name': 'CustomChatModel'}

```


```
llm.invoke("This is a foobar thing")

```


```
await llm.ainvoke("world")

```


```
llm.batch(["woof woof woof", "meow meow meow"])

```


```
await llm.abatch(["woof woof woof", "meow meow meow"])

```


```
async for token in llm.astream("hello"):
    print(token, end="|", flush=True)

```


Let’s confirm that in integrates nicely with other `LangChain` APIs.

```
from langchain_core.prompts import ChatPromptTemplate

```


```
prompt = ChatPromptTemplate.from_messages(
    [("system", "you are a bot"), ("human", "{input}")]
)

```


```
llm = CustomLLM(n=7)
chain = prompt | llm

```


```
idx = 0
async for event in chain.astream_events({"input": "hello there!"}, version="v1"):
    print(event)
    idx += 1
    if idx > 7:
        # Truncate
        break

```


```
{'event': 'on_chain_start', 'run_id': '05f24b4f-7ea3-4fb6-8417-3aa21633462f', 'name': 'RunnableSequence', 'tags': [], 'metadata': {}, 'data': {'input': {'input': 'hello there!'}}}
{'event': 'on_prompt_start', 'name': 'ChatPromptTemplate', 'run_id': '7e996251-a926-4344-809e-c425a9846d21', 'tags': ['seq:step:1'], 'metadata': {}, 'data': {'input': {'input': 'hello there!'}}}
{'event': 'on_prompt_end', 'name': 'ChatPromptTemplate', 'run_id': '7e996251-a926-4344-809e-c425a9846d21', 'tags': ['seq:step:1'], 'metadata': {}, 'data': {'input': {'input': 'hello there!'}, 'output': ChatPromptValue(messages=[SystemMessage(content='you are a bot'), HumanMessage(content='hello there!')])}}
{'event': 'on_llm_start', 'name': 'CustomLLM', 'run_id': 'a8766beb-10f4-41de-8750-3ea7cf0ca7e2', 'tags': ['seq:step:2'], 'metadata': {}, 'data': {'input': {'prompts': ['System: you are a bot\nHuman: hello there!']}}}
{'event': 'on_llm_stream', 'name': 'CustomLLM', 'run_id': 'a8766beb-10f4-41de-8750-3ea7cf0ca7e2', 'tags': ['seq:step:2'], 'metadata': {}, 'data': {'chunk': 'S'}}
{'event': 'on_chain_stream', 'run_id': '05f24b4f-7ea3-4fb6-8417-3aa21633462f', 'tags': [], 'metadata': {}, 'name': 'RunnableSequence', 'data': {'chunk': 'S'}}
{'event': 'on_llm_stream', 'name': 'CustomLLM', 'run_id': 'a8766beb-10f4-41de-8750-3ea7cf0ca7e2', 'tags': ['seq:step:2'], 'metadata': {}, 'data': {'chunk': 'y'}}
{'event': 'on_chain_stream', 'run_id': '05f24b4f-7ea3-4fb6-8417-3aa21633462f', 'tags': [], 'metadata': {}, 'name': 'RunnableSequence', 'data': {'chunk': 'y'}}

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

*   ☐ Make sure to invoke the `on_llm_new_token` callback
*   ☐ `on_llm_new_token` is invoked BEFORE yielding the chunk

Stop Token Behavior:

*   ☐ Stop token should be respected
*   ☐ Stop token should be INCLUDED as part of the response

Secret API Keys:

*   ☐ If your model connects to an API it will likely accept API keys as part of its initialization. Use Pydantic’s `SecretStr` type for secrets, so they don’t get accidentally printed out when folks print the model.