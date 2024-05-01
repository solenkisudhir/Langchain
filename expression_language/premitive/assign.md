# Assign: Add values to state | 🦜️🔗 LangChain
Adding values to chain state
----------------------------

The `RunnablePassthrough.assign(...)` static method takes an input value and adds the extra arguments passed to the assign function.

This is useful when additively creating a dictionary to use as input to a later step, which is a common LCEL pattern.

Here’s an example:

```
%pip install --upgrade --quiet langchain langchain-openai

```


```
WARNING: You are using pip version 22.0.4; however, version 24.0 is available.
You should consider upgrading via the '/Users/jacoblee/.pyenv/versions/3.10.5/bin/python -m pip install --upgrade pip' command.
Note: you may need to restart the kernel to use updated packages.

```


```
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

runnable = RunnableParallel(
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    modified=lambda x: x["num"] + 1,
)

runnable.invoke({"num": 1})

```


```
{'extra': {'num': 1, 'mult': 3}, 'modified': 2}

```


Let’s break down what’s happening here.

*   The input to the chain is `{"num": 1}`. This is passed into a `RunnableParallel`, which invokes the runnables it is passed in parallel with that input.
*   The value under the `extra` key is invoked. `RunnablePassthrough.assign()` keeps the original keys in the input dict (`{"num": 1}`), and assigns a new key called `mult`. The value is `lambda x: x["num"] * 3)`, which is `3`. Thus, the result is `{"num": 1, "mult": 3}`.
*   `{"num": 1, "mult": 3}` is returned to the `RunnableParallel` call, and is set as the value to the key `extra`.
*   At the same time, the `modified` key is called. The result is `2`, since the lambda extracts a key called `"num"` from its input and adds one.

Thus, the result is `{'extra': {'num': 1, 'mult': 3}, 'modified': 2}`.

Streaming[​](#streaming "Direct link to Streaming")
---------------------------------------------------

One nice feature of this method is that it allows values to pass through as soon as they are available. To show this off, we’ll use `RunnablePassthrough.assign()` to immediately return source docs in a retrieval chain:

```
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()

generation_chain = prompt | model | StrOutputParser()

retrieval_chain = {
    "context": retriever,
    "question": RunnablePassthrough(),
} | RunnablePassthrough.assign(output=generation_chain)

stream = retrieval_chain.stream("where did harrison work?")

for chunk in stream:
    print(chunk)

```


```
{'question': 'where did harrison work?'}
{'context': [Document(page_content='harrison worked at kensho')]}
{'output': ''}
{'output': 'H'}
{'output': 'arrison'}
{'output': ' worked'}
{'output': ' at'}
{'output': ' Kens'}
{'output': 'ho'}
{'output': '.'}
{'output': ''}

```


We can see that the first chunk contains the original `"question"` since that is immediately available. The second chunk contains `"context"` since the retriever finishes second. Finally, the output from the `generation_chain` streams in chunks as soon as it is available.