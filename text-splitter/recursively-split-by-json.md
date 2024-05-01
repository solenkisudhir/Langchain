# Recursively split JSON | 🦜️🔗 LangChain
This json splitter traverses json data depth first and builds smaller json chunks. It attempts to keep nested json objects whole but will split them if needed to keep chunks between a min\_chunk\_size and the max\_chunk\_size. If the value is not a nested json, but rather a very large string the string will not be split. If you need a hard cap on the chunk size considder following this with a Recursive Text splitter on those chunks. There is an optional pre-processing step to split lists, by first converting them to json (dict) and then splitting them as such.

1.  How the text is split: json value.
2.  How the chunk size is measured: by number of characters.

```
%pip install -qU langchain-text-splitters

```


```
import json

import requests

```


```
# This is a large nested json object and will be loaded as a python dict
json_data = requests.get("https://api.smith.langchain.com/openapi.json").json()

```


```
from langchain_text_splitters import RecursiveJsonSplitter

```


```
splitter = RecursiveJsonSplitter(max_chunk_size=300)

```


```
# Recursively split json data - If you need to access/manipulate the smaller json chunks
json_chunks = splitter.split_json(json_data=json_data)

```


```
# The splitter can also output documents
docs = splitter.create_documents(texts=[json_data])

# or a list of strings
texts = splitter.split_text(json_data=json_data)

print(texts[0])
print(texts[1])

```


```
{"openapi": "3.0.2", "info": {"title": "LangChainPlus", "version": "0.1.0"}, "paths": {"/sessions/{session_id}": {"get": {"tags": ["tracer-sessions"], "summary": "Read Tracer Session", "description": "Get a specific session.", "operationId": "read_tracer_session_sessions__session_id__get"}}}}
{"paths": {"/sessions/{session_id}": {"get": {"parameters": [{"required": true, "schema": {"title": "Session Id", "type": "string", "format": "uuid"}, "name": "session_id", "in": "path"}, {"required": false, "schema": {"title": "Include Stats", "type": "boolean", "default": false}, "name": "include_stats", "in": "query"}, {"required": false, "schema": {"title": "Accept", "type": "string"}, "name": "accept", "in": "header"}]}}}}

```


```
# Let's look at the size of the chunks
print([len(text) for text in texts][:10])

# Reviewing one of these chunks that was bigger we see there is a list object there
print(texts[1])

```


```
[293, 431, 203, 277, 230, 194, 162, 280, 223, 193]
{"paths": {"/sessions/{session_id}": {"get": {"parameters": [{"required": true, "schema": {"title": "Session Id", "type": "string", "format": "uuid"}, "name": "session_id", "in": "path"}, {"required": false, "schema": {"title": "Include Stats", "type": "boolean", "default": false}, "name": "include_stats", "in": "query"}, {"required": false, "schema": {"title": "Accept", "type": "string"}, "name": "accept", "in": "header"}]}}}}

```


```
# The json splitter by default does not split lists
# the following will preprocess the json and convert list to dict with index:item as key:val pairs
texts = splitter.split_text(json_data=json_data, convert_lists=True)

```


```
# Let's look at the size of the chunks. Now they are all under the max
print([len(text) for text in texts][:10])

```


```
[293, 431, 203, 277, 230, 194, 162, 280, 223, 193]

```


```
# The list has been converted to a dict, but retains all the needed contextual information even if split into many chunks
print(texts[1])

```


```
{"paths": {"/sessions/{session_id}": {"get": {"parameters": [{"required": true, "schema": {"title": "Session Id", "type": "string", "format": "uuid"}, "name": "session_id", "in": "path"}, {"required": false, "schema": {"title": "Include Stats", "type": "boolean", "default": false}, "name": "include_stats", "in": "query"}, {"required": false, "schema": {"title": "Accept", "type": "string"}, "name": "accept", "in": "header"}]}}}}

```


```
# We can also look at the documents
docs[1]

```


```
Document(page_content='{"paths": {"/sessions/{session_id}": {"get": {"parameters": [{"required": true, "schema": {"title": "Session Id", "type": "string", "format": "uuid"}, "name": "session_id", "in": "path"}, {"required": false, "schema": {"title": "Include Stats", "type": "boolean", "default": false}, "name": "include_stats", "in": "query"}, {"required": false, "schema": {"title": "Accept", "type": "string"}, "name": "accept", "in": "header"}]}}}}')

```
