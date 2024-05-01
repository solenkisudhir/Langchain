# Quickstart | 🦜️🔗 LangChain
Chat models are a variation on language models. While chat models use language models under the hood, the interface they use is a bit different. Rather than using a “text in, text out” API, they use an interface where “chat messages” are the inputs and outputs.

If you’d prefer not to set an environment variable you can pass the key in directly via the api key arg named parameter when initiating the chat model class:

The chat model interface is based around messages rather than raw text. The types of messages currently supported in LangChain are `AIMessage`, `HumanMessage`, `SystemMessage`, `FunctionMessage` and `ChatMessage` – `ChatMessage` takes in an arbitrary role parameter. Most of the time, you’ll just be dealing with `HumanMessage`, `AIMessage`, and `SystemMessage`

Chat models accept `List[BaseMessage]` as inputs, or objects which can be coerced to messages, including `str` (converted to `HumanMessage`) and `PromptValue`.

```
AIMessage(content="The purpose of model regularization is to prevent overfitting in machine learning models. Overfitting occurs when a model becomes too complex and starts to fit the noise in the training data, leading to poor generalization on unseen data. Regularization techniques introduce additional constraints or penalties to the model's objective function, discouraging it from becoming overly complex and promoting simpler and more generalizable models. Regularization helps to strike a balance between fitting the training data well and avoiding overfitting, leading to better performance on new, unseen data.")

```


```
The purpose of model regularization is to prevent overfitting and improve the generalization of a machine learning model. Overfitting occurs when a model is too complex and learns the noise or random variations in the training data, which leads to poor performance on new, unseen data. Regularization techniques introduce additional constraints or penalties to the model's learning process, discouraging it from fitting the noise and reducing the complexity of the model. This helps to improve the model's ability to generalize well and make accurate predictions on unseen data.

```


```
[AIMessage(content="The purpose of model regularization is to prevent overfitting in machine learning models. Overfitting occurs when a model becomes too complex and starts to learn the noise or random fluctuations in the training data, rather than the underlying patterns or relationships. Regularization techniques add a penalty term to the model's objective function, which discourages the model from becoming too complex and helps it generalize better to new, unseen data. This improves the model's ability to make accurate predictions on new data by reducing the variance and increasing the model's overall performance.")]

```


```
AIMessage(content='The purpose of model regularization is to prevent overfitting in machine learning models. Overfitting occurs when a model becomes too complex and starts to memorize the training data instead of learning general patterns and relationships. This leads to poor performance on new, unseen data.\n\nRegularization techniques introduce additional constraints or penalties to the model during training, discouraging it from becoming overly complex. This helps to strike a balance between fitting the training data well and generalizing to new data. Regularization techniques can include adding a penalty term to the loss function, such as L1 or L2 regularization, or using techniques like dropout or early stopping. By regularizing the model, it encourages it to learn the most relevant features and reduce the impact of noise or outliers in the data.')

```


```
The purpose of model regularization is to prevent overfitting in machine learning models. Overfitting occurs when a model becomes too complex and starts to memorize the training data instead of learning the underlying patterns. Regularization techniques help in reducing the complexity of the model by adding a penalty to the loss function. This penalty encourages the model to have smaller weights or fewer features, making it more generalized and less prone to overfitting. The goal is to find the right balance between fitting the training data well and being able to generalize well to unseen data.

```


```
RunLogPatch({'op': 'replace',
  'path': '',
  'value': {'final_output': None,
            'id': '754c4143-2348-46c4-ad2b-3095913084c6',
            'logs': {},
            'streamed_output': []}})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='The')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' purpose')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' of')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' model')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' regularization')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' is')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' to')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' prevent')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' a')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' machine')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' learning')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' model')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' from')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' over')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='fit')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='ting')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' the')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' training')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' data')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' and')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' improve')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' its')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' general')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='ization')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' ability')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='.')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' Over')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='fit')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='ting')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' occurs')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' when')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' a')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' model')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' becomes')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' too')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' complex')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' and')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' learns')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' to')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' fit')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' the')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' noise')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' or')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' random')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' fluctuations')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' in')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' the')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' training')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' data')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=',')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' instead')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' of')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' capturing')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' the')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' underlying')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' patterns')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' and')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' relationships')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='.')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' Regular')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='ization')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' techniques')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' introduce')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' a')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' penalty')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' term')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' to')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' the')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' model')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content="'s")})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' objective')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' function')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=',')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' which')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' discour')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='ages')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' the')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' model')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' from')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' becoming')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' too')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' complex')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='.')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' This')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' helps')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' to')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' control')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' the')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' model')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content="'s")})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' complexity')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' and')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' reduces')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' the')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' risk')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' of')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' over')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='fit')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='ting')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=',')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' leading')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' to')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' better')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' performance')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' on')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' unseen')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content=' data')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='.')})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': AIMessageChunk(content='')})
RunLogPatch({'op': 'replace',
  'path': '/final_output',
  'value': {'generations': [[{'generation_info': {'finish_reason': 'stop'},
                              'message': AIMessageChunk(content="The purpose of model regularization is to prevent a machine learning model from overfitting the training data and improve its generalization ability. Overfitting occurs when a model becomes too complex and learns to fit the noise or random fluctuations in the training data, instead of capturing the underlying patterns and relationships. Regularization techniques introduce a penalty term to the model's objective function, which discourages the model from becoming too complex. This helps to control the model's complexity and reduces the risk of overfitting, leading to better performance on unseen data."),
                              'text': 'The purpose of model regularization is '
                                      'to prevent a machine learning model '
                                      'from overfitting the training data and '
                                      'improve its generalization ability. '
                                      'Overfitting occurs when a model becomes '
                                      'too complex and learns to fit the noise '
                                      'or random fluctuations in the training '
                                      'data, instead of capturing the '
                                      'underlying patterns and relationships. '
                                      'Regularization techniques introduce a '
                                      "penalty term to the model's objective "
                                      'function, which discourages the model '
                                      'from becoming too complex. This helps '
                                      "to control the model's complexity and "
                                      'reduces the risk of overfitting, '
                                      'leading to better performance on unseen '
                                      'data.'}]],
            'llm_output': None,
            'run': None}})

```


All `ChatModel`s come with built-in LangSmith tracing. Just set the following environment variables:

and any `ChatModel` invocation (whether it’s nested in a chain or not) will automatically be traced. A trace will include inputs, outputs, latency, token usage, invocation params, environment params, and more. See an example here: [https://smith.langchain.com/public/a54192ae-dd5c-4f7a-88d1-daa1eaba1af7/r](https://smith.langchain.com/public/a54192ae-dd5c-4f7a-88d1-daa1eaba1af7/r).

In LangSmith you can then provide feedback for any trace, compile annotated datasets for evals, debug performance in the playground, and more.

For convenience you can also treat chat models as callables. You can get chat completions by passing one or more messages to the chat model. The response will be a message.

```
AIMessage(content="J'adore la programmation.")

```


OpenAI’s chat model supports multiple messages as input. See [here](https://platform.openai.com/docs/guides/chat/chat-vs-completions) for more information. Here is an example of sending a system and user message to the chat model:

```
AIMessage(content="J'adore la programmation.")

```


You can go one step further and generate completions for multiple sets of messages using `generate`. This returns an `LLMResult` with an additional `message` parameter. This will include additional information about each generation beyond the returned message (e.g. the finish reason) and additional information about the full API call (e.g. total tokens used).

```
LLMResult(generations=[[ChatGeneration(text="J'adore programmer.", generation_info={'finish_reason': 'stop'}, message=AIMessage(content="J'adore programmer."))], [ChatGeneration(text="J'adore l'intelligence artificielle.", generation_info={'finish_reason': 'stop'}, message=AIMessage(content="J'adore l'intelligence artificielle."))]], llm_output={'token_usage': {'prompt_tokens': 53, 'completion_tokens': 18, 'total_tokens': 71}, 'model_name': 'gpt-3.5-turbo'}, run=[RunInfo(run_id=UUID('077917a9-026c-47c4-b308-77b37c3a3bfa')), RunInfo(run_id=UUID('0a70a0bf-c599-4f51-932a-c7d42202c984'))])

```


```
{'token_usage': {'prompt_tokens': 53,
  'completion_tokens': 18,
  'total_tokens': 71},
 'model_name': 'gpt-3.5-turbo'}

```
