# Caching | 🦜️🔗 LangChain
LangChain provides an optional caching layer for chat models. This is useful for two reasons:

It can save you money by reducing the number of API calls you make to the LLM provider, if you’re often requesting the same completion multiple times. It can speed up your application by reducing the number of API calls you make to the LLM provider.

*   OpenAI
*   Anthropic
*   Google
*   Cohere
*   FireworksAI
*   MistralAI
*   TogetherAI

##### Install dependencies

```
pip install -qU langchain-openai

```


##### Set environment variables

```
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

```


```
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

```


```
# <!-- ruff: noqa: F821 -->
from langchain.globals import set_llm_cache

```


In Memory Cache[​](#in-memory-cache "Direct link to In Memory Cache")
---------------------------------------------------------------------

```
%%time
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())

# The first time, it is not yet in cache, so it should take longer
llm.predict("Tell me a joke")

```


```
CPU times: user 17.7 ms, sys: 9.35 ms, total: 27.1 ms
Wall time: 801 ms

```


```
"Sure, here's a classic one for you:\n\nWhy don't scientists trust atoms?\n\nBecause they make up everything!"

```


```
%%time
# The second time it is, so it goes faster
llm.predict("Tell me a joke")

```


```
CPU times: user 1.42 ms, sys: 419 µs, total: 1.83 ms
Wall time: 1.83 ms

```


```
"Sure, here's a classic one for you:\n\nWhy don't scientists trust atoms?\n\nBecause they make up everything!"

```


SQLite Cache[​](#sqlite-cache "Direct link to SQLite Cache")
------------------------------------------------------------

```
# We can do the same thing with a SQLite cache
from langchain.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

```


```
%%time
# The first time, it is not yet in cache, so it should take longer
llm.predict("Tell me a joke")

```


```
CPU times: user 23.2 ms, sys: 17.8 ms, total: 40.9 ms
Wall time: 592 ms

```


```
"Sure, here's a classic one for you:\n\nWhy don't scientists trust atoms?\n\nBecause they make up everything!"

```


```
%%time
# The second time it is, so it goes faster
llm.predict("Tell me a joke")

```


```
CPU times: user 5.61 ms, sys: 22.5 ms, total: 28.1 ms
Wall time: 47.5 ms

```


```
"Sure, here's a classic one for you:\n\nWhy don't scientists trust atoms?\n\nBecause they make up everything!"

```


* * *

#### Help us out by providing feedback on this documentation page: