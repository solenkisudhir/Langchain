# Quick Start | 🦜️🔗 LangChain
Large Language Models (LLMs) are a core component of LangChain. LangChain does not serve its own LLMs, but rather provides a standard interface for interacting with many different LLMs.

There are lots of LLM providers (OpenAI, Cohere, Hugging Face, etc) - the `LLM` class is designed to provide a standard interface for all of them.

In this walkthrough we’ll work with an OpenAI LLM wrapper, although the functionalities highlighted are generic for all LLM types.

Accessing the API requires an API key, which you can get by creating an account and heading [here](https://platform.openai.com/account/api-keys). Once we have a key we’ll want to set it as an environment variable by running:

If you’d prefer not to set an environment variable you can pass the key in directly via the `api_key` named parameter when initiating the OpenAI LLM class:

LLMs accept **strings** as inputs, or objects which can be coerced to string prompts, including `List[BaseMessage]` and `PromptValue`.

```
'\n\n1. The Phillips Curve Theory: This suggests that there is an inverse relationship between unemployment and inflation, meaning that when unemployment is low, inflation will be higher, and when unemployment is high, inflation will be lower.\n\n2. The Monetarist Theory: This theory suggests that the relationship between unemployment and inflation is weak, and that changes in the money supply are more important in determining inflation.\n\n3. The Resource Utilization Theory: This suggests that when unemployment is low, firms are able to raise wages and prices in order to take advantage of the increased demand for their products and services. This leads to higher inflation.'

```


```


1. The Phillips Curve Theory: This theory states that there is an inverse relationship between unemployment and inflation. As unemployment decreases, inflation increases and vice versa.

2. The Cost-Push Inflation Theory: This theory suggests that an increase in unemployment leads to a decrease in aggregate demand, which causes prices to go up due to a decrease in supply.

3. The Wage-Push Inflation Theory: This theory states that when unemployment is low, wages tend to increase due to competition for labor, which causes prices to rise.

4. The Monetarist Theory: This theory states that there is no direct relationship between unemployment and inflation, but rather, an increase in the money supply leads to inflation, which can be caused by an increase in unemployment.

```


```
['\n\n1. The Phillips Curve Theory: This theory suggests that there is an inverse relationship between unemployment and inflation, meaning that when unemployment decreases, inflation rises, and when unemployment increases, inflation decreases. This theory is based on the idea that when the economy is doing well, there is more demand for goods and services, causing prices to increase.\n\n2. The Cost-Push Theory: This theory suggests that when the cost of production increases, it leads to higher prices and lower output. This can lead to higher unemployment and eventually higher inflation.\n\n3. The Demand-Pull Theory: This theory suggests that when demand for goods and services increases, it leads to higher prices and eventually higher inflation. This can lead to higher unemployment as businesses cannot keep up with the higher demand.\n\n4. The Structural Unemployment Theory: This theory suggests that when there is a mismatch between the skills of the unemployed and the skills required in the job market, it leads to higher unemployment and eventually higher inflation.']

```


```
'\n\n1. Phillips Curve Theory: This theory states that there is an inverse relationship between inflation and unemployment. As unemployment decreases, inflation increases, and vice versa.\n\n2. Cost-Push Theory: This theory suggests that inflation is caused by rising costs, which can be caused by an increase in unemployment. As unemployment rises, businesses are unable to keep up with demand and have to raise prices to compensate.\n\n3. Demand-Pull Theory: This theory suggests that inflation occurs when demand exceeds supply. As unemployment increases, demand for goods and services decreases, leading to a decrease in inflation.\n\n4. Monetary Theory: This theory suggests that the money supply and inflation are related to unemployment. When the money supply increases, prices increase, leading to an increase in inflation. If unemployment is high, then the money supply increases, leading to an increase in inflation.'

```


```


1. Phillips Curve Theory: This theory suggests that there is an inverse relationship between unemployment and inflation, meaning that when unemployment is low, inflation rises and vice versa.

2. Cost-Push Theory: This theory suggests that inflation is caused by rising costs of production, such as wages, raw materials, and energy. It states that when costs increase, firms must pass these costs onto the consumer, thus raising the price of goods and services and leading to inflation.

3. Demand-Pull Theory: This theory suggests that inflation is caused by an increase in demand for goods and services, leading to a rise in prices. It suggests that when unemployment is low, people have more money to spend and this increased demand pushes up prices.

4. Monetarist Theory: This theory states that inflation is caused by an increase in the money supply. It suggests that when the money supply increases, people have more money to spend, leading to higher prices.

```


```
['\n\n1. The Phillips Curve Theory: This theory states that there is an inverse relationship between unemployment and inflation. When unemployment is low, wages increase, leading to higher prices and overall inflation.\n\n2. The Cost-Push Theory: This theory states that inflation is caused by increases in the costs of production, such as wages, goods, and services. When the cost of production increases, the prices of goods and services must also increase, leading to inflation.\n\n3. The Demand Pull Theory: This theory states that inflation is caused by an increase in aggregate demand for goods and services. When the demand is high, prices must increase in order to meet the demand. This leads to inflation.\n\n4. The Structural Unemployment Theory: This theory states that when unemployment is high, there is an excess supply of labor. This excess supply of labor can result in lower wages, which can cause inflation as people are willing to accept lower wages for the same amount of work.']

```


```
RunLogPatch({'op': 'replace',
  'path': '',
  'value': {'final_output': None,
            'id': 'baf410ad-618e-44db-93c8-809da4e3ed44',
            'logs': {},
            'streamed_output': []}})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': '\n'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': '\n'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': '1'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': '.'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' The'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' Phillips'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' Curve'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ':'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' This'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' theory'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' suggests'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' that'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' there'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' is'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' an'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' inverse'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' relationship'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' between'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' unemployment and'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' inflation'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': '.'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' When'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' unemployment'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' is'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' low'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ','})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' inflation'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' tends'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' to'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' be'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' high'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ','})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' and'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' when'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' unemployment'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' is'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' high'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ','})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' inflation'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' tends'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' to'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' be'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' low'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': '.'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' '})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': '\n'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': '\n'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': '2'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': '.'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' The'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' NA'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': 'IR'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': 'U'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' Theory'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ':'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' This'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' theory'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' suggests'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' that there is'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' a'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' natural'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' rate'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' of'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' unemployment'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ','})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' also'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' known'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' as'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' the'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' Non'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': '-'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': 'Ac'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': 'celer'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': 'ating'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' In'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': 'flation'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' Rate'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' of'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' Unemployment'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' ('})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': 'NA'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': 'IR'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': 'U'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ').'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' According'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' to'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' this'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' theory'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ','})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' when'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' unemployment'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' is'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' below'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' the'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' NA'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': 'IR'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': 'U'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ','})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' then'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' inflation'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' will'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' increase'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ','})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' and'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' when'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' unemployment'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' is'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' above'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' the'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' NA'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': 'IR'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': 'U'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ','})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' then'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' inflation'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' will'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' decrease'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': '.'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': '\n'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': '\n'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': '3'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': '.'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' The'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' Cost'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': '-'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': 'Push'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' In'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': 'flation'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' Theory'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ':'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' This'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' theory'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' suggests'})
RunLogPatch({'op': 'add',
  'path': '/streamed_output/-',
  'value': ' that high unemployment'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' leads'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' to'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' higher'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' wages'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ','})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' which'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' in'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' turn'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' leads'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' to'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' higher'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' prices'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ' and higher inflation'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': '.'})
RunLogPatch({'op': 'add', 'path': '/streamed_output/-', 'value': ''})
RunLogPatch({'op': 'replace',
  'path': '/final_output',
  'value': {'generations': [[{'generation_info': {'finish_reason': 'stop',
                                                  'logprobs': None},
                              'text': '\n'
                                      '\n'
                                      '1. The Phillips Curve: This theory '
                                      'suggests that there is an inverse '
                                      'relationship between unemployment and '
                                      'inflation. When unemployment is low, '
                                      'inflation tends to be high, and when '
                                      'unemployment is high, inflation tends '
                                      'to be low. \n'
                                      '\n'
                                      '2. The NAIRU Theory: This theory '
                                      'suggests that there is a natural rate '
                                      'of unemployment, also known as the '
                                      'Non-Accelerating Inflation Rate of '
                                      'Unemployment (NAIRU). According to this '
                                      'theory, when unemployment is below the '
                                      'NAIRU, then inflation will increase, '
                                      'and when unemployment is above the '
                                      'NAIRU, then inflation will decrease.\n'
                                      '\n'
                                      '3. The Cost-Push Inflation Theory: This '
                                      'theory suggests that high unemployment '
                                      'leads to higher wages, which in turn '
                                      'leads to higher prices and higher '
                                      'inflation.'}]],
            'llm_output': None,
            'run': None}})

```


All `LLM`s come with built-in LangSmith tracing. Just set the following environment variables:

and any `LLM` invocation (whether it’s nested in a chain or not) will automatically be traced. A trace will include inputs, outputs, latency, token usage, invocation params, environment params, and more. See an example here: [https://smith.langchain.com/public/7924621a-ff58-4b1c-a2a2-035a354ef434/r](https://smith.langchain.com/public/7924621a-ff58-4b1c-a2a2-035a354ef434/r).

In LangSmith you can then provide feedback for any trace, compile annotated datasets for evals, debug performance in the playground, and more.