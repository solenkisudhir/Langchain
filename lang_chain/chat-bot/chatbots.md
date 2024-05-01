# Chatbots | 🦜️🔗 LangChain
Overview[​](#overview "Direct link to Overview")
------------------------------------------------

Chatbots are one of the most popular use-cases for LLMs. The core features of chatbots are that they can have long-running, stateful conversations and can answer user questions using relevant information.

Architectures[​](#architectures "Direct link to Architectures")
---------------------------------------------------------------

Designing a chatbot involves considering various techniques with different benefits and tradeoffs depending on what sorts of questions you expect it to handle.

For example, chatbots commonly use [retrieval-augmented generation](https://python.langchain.com/docs/use_cases/question_answering/), or RAG, over private data to better answer domain-specific questions. You also might choose to route between multiple data sources to ensure it only uses the most topical context for final question answering, or choose to use a more specialized type of chat history or memory than just passing messages back and forth.

![Image description](https://python.langchain.com/assets/images/chat_use_case-eb8a4883931d726e9f23628a0d22e315.png)

Optimizations like this can make your chatbot more powerful, but add latency and complexity. The aim of this guide is to give you an overview of how to implement various features and help you tailor your chatbot to your particular use-case.

Table of contents[​](#table-of-contents "Direct link to Table of contents")
---------------------------------------------------------------------------

*   [Quickstart](https://python.langchain.com/docs/use_cases/chatbots/quickstart/): We recommend starting here. Many of the following guides assume you fully understand the architecture shown in the Quickstart.
*   [Memory management](https://python.langchain.com/docs/use_cases/chatbots/memory_management/): This section covers various strategies your chatbot can use to handle information from previous conversation turns.
*   [Retrieval](https://python.langchain.com/docs/use_cases/chatbots/retrieval/): This section covers how to enable your chatbot to use outside data sources as context.
*   [Tool usage](https://python.langchain.com/docs/use_cases/chatbots/tool_usage/): This section covers how to turn your chatbot into a conversational agent by adding the ability to interact with other systems and APIs using tools.