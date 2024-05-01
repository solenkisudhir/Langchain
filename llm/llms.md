# LLMs | 🦜️🔗 LangChain
Large Language Models (LLMs) are a core component of LangChain. LangChain does not serve its own LLMs, but rather provides a standard interface for interacting with many different LLMs. To be specific, this interface is one that takes as input a string and returns a string.

There are lots of LLM providers (OpenAI, Cohere, Hugging Face, etc) - the `LLM` class is designed to provide a standard interface for all of them.

[Quick Start](https://python.langchain.com/docs/modules/model_io/llms/quick_start/)[​](#quick-start "Direct link to quick-start")
---------------------------------------------------------------------------------------------------------------------------------

Check out [this quick start](https://python.langchain.com/docs/modules/model_io/llms/quick_start/) to get an overview of working with LLMs, including all the different methods they expose

[Integrations](https://python.langchain.com/docs/integrations/llms/)[​](#integrations "Direct link to integrations")
--------------------------------------------------------------------------------------------------------------------

For a full list of all LLM integrations that LangChain provides, please go to the [Integrations page](https://python.langchain.com/docs/integrations/llms/)

How-To Guides[​](#how-to-guides "Direct link to How-To Guides")
---------------------------------------------------------------

We have several how-to guides for more advanced usage of LLMs. This includes:

*   [How to write a custom LLM class](https://python.langchain.com/docs/modules/model_io/llms/custom_llm/)
*   [How to cache LLM responses](https://python.langchain.com/docs/modules/model_io/llms/llm_caching/)
*   [How to stream responses from an LLM](https://python.langchain.com/docs/modules/model_io/llms/streaming_llm/)
*   [How to track token usage in an LLM call](https://python.langchain.com/docs/modules/model_io/llms/token_usage_tracking/)