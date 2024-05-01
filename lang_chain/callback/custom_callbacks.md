# Custom callback handlers | 🦜️🔗 LangChain
To create a custom callback handler we need to determine the [event(s)](https://python.langchain.com/docs/modules/callbacks/) we want our callback handler to handle as well as what we want our callback handler to do when the event is triggered. Then all we need to do is attach the callback handler to the object either as a constructer callback or a request callback (see [callback types](https://python.langchain.com/docs/modules/callbacks/)).

In the example below, we’ll implement streaming with a custom handler.

In our custom callback handler `MyCustomHandler`, we implement the `on_llm_new_token` to print the token we have just received. We then attach our custom handler to the model object as a constructor callback.

```
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"My custom handler, token: {token}")


prompt = ChatPromptTemplate.from_messages(["Tell me a joke about {animal}"])

# To enable streaming, we pass in `streaming=True` to the ChatModel constructor
# Additionally, we pass in our custom handler as a list to the callbacks parameter
model = ChatOpenAI(streaming=True, callbacks=[MyCustomHandler()])

chain = prompt | model

response = chain.invoke({"animal": "bears"})

```


```
My custom handler, token: 
My custom handler, token: Why
My custom handler, token:  do
My custom handler, token:  bears
My custom handler, token:  have
My custom handler, token:  hairy
My custom handler, token:  coats
My custom handler, token: ?


My custom handler, token: F
My custom handler, token: ur
My custom handler, token:  protection
My custom handler, token: !
My custom handler, token: 

```
