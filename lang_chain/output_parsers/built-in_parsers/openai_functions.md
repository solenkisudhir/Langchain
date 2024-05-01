# OpenAI Functions | 🦜️🔗 LangChain
These output parsers use OpenAI function calling to structure its outputs. This means they are only usable with models that support function calling. There are a few different variants:

```
{'setup': "Why don't scientists trust atoms?",
 'punchline': 'Because they make up everything!'}

```


```
{}
{'setup': ''}
{'setup': 'Why'}
{'setup': 'Why don'}
{'setup': "Why don't"}
{'setup': "Why don't scientists"}
{'setup': "Why don't scientists trust"}
{'setup': "Why don't scientists trust atoms"}
{'setup': "Why don't scientists trust atoms?", 'punchline': ''}
{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because'}
{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they'}
{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make'}
{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up'}
{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything'}
{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}

```


This merely extracts a single key from the returned response. This is useful for when you want to return a list of things.

```
[{'setup': "Why don't scientists trust atoms?",
  'punchline': 'Because they make up everything!'},
 {'setup': 'Why did the scarecrow win an award?',
  'punchline': 'Because he was outstanding in his field!'}]

```


```
[]
[{}]
[{'setup': ''}]
[{'setup': 'Why'}]
[{'setup': 'Why don'}]
[{'setup': "Why don't"}]
[{'setup': "Why don't scientists"}]
[{'setup': "Why don't scientists trust"}]
[{'setup': "Why don't scientists trust atoms"}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': ''}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}, {}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}, {'setup': ''}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}, {'setup': 'Why'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}, {'setup': 'Why did'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}, {'setup': 'Why did the'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}, {'setup': 'Why did the scare'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}, {'setup': 'Why did the scarecrow'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}, {'setup': 'Why did the scarecrow win'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}, {'setup': 'Why did the scarecrow win an'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}, {'setup': 'Why did the scarecrow win an award'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}, {'setup': 'Why did the scarecrow win an award?', 'punchline': ''}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}, {'setup': 'Why did the scarecrow win an award?', 'punchline': 'Because'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}, {'setup': 'Why did the scarecrow win an award?', 'punchline': 'Because he'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}, {'setup': 'Why did the scarecrow win an award?', 'punchline': 'Because he was'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}, {'setup': 'Why did the scarecrow win an award?', 'punchline': 'Because he was outstanding'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}, {'setup': 'Why did the scarecrow win an award?', 'punchline': 'Because he was outstanding in'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}, {'setup': 'Why did the scarecrow win an award?', 'punchline': 'Because he was outstanding in his'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}, {'setup': 'Why did the scarecrow win an award?', 'punchline': 'Because he was outstanding in his field'}]
[{'setup': "Why don't scientists trust atoms?", 'punchline': 'Because they make up everything!'}, {'setup': 'Why did the scarecrow win an award?', 'punchline': 'Because he was outstanding in his field!'}]

```


This builds on top of `JsonOutputFunctionsParser` but passes the results to a Pydantic Model. This allows for further validation should you choose.

```
Joke(setup="Why don't scientists trust atoms?", punchline='Because they make up everything!')

```
