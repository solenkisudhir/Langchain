# Get log probabilities | 🦜️🔗 LangChain
Certain chat models can be configured to return token-level log probabilities. This guide walks through how to get logprobs for a number of models.

For the OpenAI API to return log probabilities we need to configure the `logprobs=True` param

The logprobs are included on each output Message as part of the `response_metadata`:

```
[{'token': 'As',
  'bytes': [65, 115],
  'logprob': -1.5358024,
  'top_logprobs': []},
 {'token': ' an',
  'bytes': [32, 97, 110],
  'logprob': -0.028062303,
  'top_logprobs': []},
 {'token': ' AI',
  'bytes': [32, 65, 73],
  'logprob': -0.009415812,
  'top_logprobs': []},
 {'token': ',', 'bytes': [44], 'logprob': -0.07371779, 'top_logprobs': []},
 {'token': ' I',
  'bytes': [32, 73],
  'logprob': -4.298773e-05,
  'top_logprobs': []}]

```


```
[]
[{'token': 'As', 'bytes': [65, 115], 'logprob': -1.7523563, 'top_logprobs': []}]
[{'token': 'As', 'bytes': [65, 115], 'logprob': -1.7523563, 'top_logprobs': []}, {'token': ' an', 'bytes': [32, 97, 110], 'logprob': -0.019908238, 'top_logprobs': []}]
[{'token': 'As', 'bytes': [65, 115], 'logprob': -1.7523563, 'top_logprobs': []}, {'token': ' an', 'bytes': [32, 97, 110], 'logprob': -0.019908238, 'top_logprobs': []}, {'token': ' AI', 'bytes': [32, 65, 73], 'logprob': -0.0093033705, 'top_logprobs': []}]
[{'token': 'As', 'bytes': [65, 115], 'logprob': -1.7523563, 'top_logprobs': []}, {'token': ' an', 'bytes': [32, 97, 110], 'logprob': -0.019908238, 'top_logprobs': []}, {'token': ' AI', 'bytes': [32, 65, 73], 'logprob': -0.0093033705, 'top_logprobs': []}, {'token': ',', 'bytes': [44], 'logprob': -0.08852102, 'top_logprobs': []}]

```
