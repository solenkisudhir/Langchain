# File logging | 🦜️🔗 LangChain
LangChain provides the `FileCallbackHandler` to write logs to a file. The `FileCallbackHandler` is similar to the [`StdOutCallbackHandler`](https://python.langchain.com/docs/modules/callbacks/), but instead of printing logs to standard output it writes logs to a file.

We see how to use the `FileCallbackHandler` in this example. Additionally we use the `StdOutCallbackHandler` to print logs to the standard output. It also uses the `loguru` library to log other outputs that are not captured by the handler.

```
from langchain_core.callbacks import FileCallbackHandler, StdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from loguru import logger

logfile = "output.log"

logger.add(logfile, colorize=True, enqueue=True)
handler_1 = FileCallbackHandler(logfile)
handler_2 = StdOutCallbackHandler()

prompt = PromptTemplate.from_template("1 + {number} = ")
model = OpenAI()

# this chain will both print to stdout (because verbose=True) and write to 'output.log'
# if verbose=False, the FileCallbackHandler will still write to 'output.log'
chain = prompt | model

response = chain.invoke({"number": 2}, {"callbacks": [handler_1, handler_2]})
logger.info(response)

```


```


> Entering new LLMChain chain...
Prompt after formatting:
1 + 2 = 

> Finished chain.

```


```
2023-06-01 18:36:38.929 | INFO     | __main__:<module>:20 - 

3

```


Now we can open the file `output.log` to see that the output has been captured.

```
%pip install --upgrade --quiet  ansi2html > /dev/null

```


```
from ansi2html import Ansi2HTMLConverter
from IPython.display import HTML, display

with open("output.log", "r") as f:
    content = f.read()

conv = Ansi2HTMLConverter()
html = conv.convert(content, full=True)

display(HTML(html))

```


```
> Entering new LLMChain chain...
Prompt after formatting:
1 + 2 = 
> Finished chain.
2023-06-01 18:36:38.929 | INFO     | __main__:<module>:20 - 
3

```
