# Installation | 🦜️🔗 LangChain
Official release[​](#official-release "Direct link to Official release")
------------------------------------------------------------------------

To install LangChain run:

*   Pip
*   Conda

This will install the bare minimum requirements of LangChain. A lot of the value of LangChain comes when integrating it with various model providers, datastores, etc. By default, the dependencies needed to do that are NOT installed. You will need to install the dependencies for specific integrations separately.

From source[​](#from-source "Direct link to From source")
---------------------------------------------------------

If you want to install from source, you can do so by cloning the repo and be sure that the directory is `PATH/TO/REPO/langchain/libs/langchain` running:

LangChain core[​](#langchain-core "Direct link to LangChain core")
------------------------------------------------------------------

The `langchain-core` package contains base abstractions that the rest of the LangChain ecosystem uses, along with the LangChain Expression Language. It is automatically installed by `langchain`, but can also be used separately. Install with:

```
pip install langchain-core

```


The `langchain-community` package contains third-party integrations. It is automatically installed by `langchain`, but can also be used separately. Install with:

```
pip install langchain-community

```


LangChain experimental[​](#langchain-experimental "Direct link to LangChain experimental")
------------------------------------------------------------------------------------------

The `langchain-experimental` package holds experimental LangChain code, intended for research and experimental uses. Install with:

```
pip install langchain-experimental

```


LangGraph[​](#langgraph "Direct link to LangGraph")
---------------------------------------------------

`langgraph` is a library for building stateful, multi-actor applications with LLMs, built on top of (and intended to be used with) LangChain. Install with:

LangServe[​](#langserve "Direct link to LangServe")
---------------------------------------------------

LangServe helps developers deploy LangChain runnables and chains as a REST API. LangServe is automatically installed by LangChain CLI. If not using LangChain CLI, install with:

```
pip install "langserve[all]"

```


for both client and server dependencies. Or `pip install "langserve[client]"` for client code, and `pip install "langserve[server]"` for server code.

LangChain CLI[​](#langchain-cli "Direct link to LangChain CLI")
---------------------------------------------------------------

The LangChain CLI is useful for working with LangChain templates and other LangServe projects. Install with:

```
pip install langchain-cli

```


LangSmith SDK[​](#langsmith-sdk "Direct link to LangSmith SDK")
---------------------------------------------------------------

The LangSmith SDK is automatically installed by LangChain. If not using LangChain, install with: