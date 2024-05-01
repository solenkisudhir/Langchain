# Mapping values to database | 🦜️🔗 LangChain
In this guide we’ll go over strategies to improve graph database query generation by mapping values from user inputs to database. When using the built-in graph chains, the LLM is aware of the graph schema, but has no information about the values of properties stored in the database. Therefore, we can introduce a new step in graph database QA system to accurately map values.

Setup[​](#setup "Direct link to Setup")
---------------------------------------

First, get required packages and set environment variables:

```
%pip install --upgrade --quiet  langchain langchain-community langchain-openai neo4j

```


```
Note: you may need to restart the kernel to use updated packages.

```


We default to OpenAI models in this guide, but you can swap them out for the model provider of your choice.

```
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

# Uncomment the below to use LangSmith. Not required.
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
# os.environ["LANGCHAIN_TRACING_V2"] = "true"

```


Next, we need to define Neo4j credentials. Follow [these installation steps](https://neo4j.com/docs/operations-manual/current/installation/) to set up a Neo4j database.

```
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"

```


The below example will create a connection with a Neo4j database and will populate it with example data about movies and their actors.

```
from langchain_community.graphs import Neo4jGraph

graph = Neo4jGraph()

# Import movie information

movies_query = """
LOAD CSV WITH HEADERS FROM 
'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv'
AS row
MERGE (m:Movie {id:row.movieId})
SET m.released = date(row.released),
    m.title = row.title,
    m.imdbRating = toFloat(row.imdbRating)
FOREACH (director in split(row.director, '|') | 
    MERGE (p:Person {name:trim(director)})
    MERGE (p)-[:DIRECTED]->(m))
FOREACH (actor in split(row.actors, '|') | 
    MERGE (p:Person {name:trim(actor)})
    MERGE (p)-[:ACTED_IN]->(m))
FOREACH (genre in split(row.genres, '|') | 
    MERGE (g:Genre {name:trim(genre)})
    MERGE (m)-[:IN_GENRE]->(g))
"""

graph.query(movies_query)

```


Detecting entities in the user input[​](#detecting-entities-in-the-user-input "Direct link to Detecting entities in the user input")
------------------------------------------------------------------------------------------------------------------------------------

We have to extract the types of entities/values we want to map to a graph database. In this example, we are dealing with a movie graph, so we can map movies and people to the database.

```
from typing import List, Optional

from langchain.chains.openai_functions import create_structured_output_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person or movies appearing in the text",
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting person and movies from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)


entity_chain = create_structured_output_chain(Entities, llm, prompt)

```


```
/Users/tomazbratanic/anaconda3/lib/python3.11/site-packages/langchain_core/_api/deprecation.py:117: LangChainDeprecationWarning: The function `create_structured_output_chain` was deprecated in LangChain 0.1.1 and will be removed in 0.2.0. Use create_structured_output_runnable instead.
  warn_deprecated(

```


We can test the entity extraction chain.

```
entities = entity_chain.invoke({"question": "Who played in Casino movie?"})
entities

```


```
{'question': 'Who played in Casino movie?',
 'function': Entities(names=['Casino'])}

```


We will utilize a simple `CONTAINS` clause to match entities to database. In practice, you might want to use a fuzzy search or a fulltext index to allow for minor misspellings.

```
match_query = """MATCH (p:Person|Movie)
WHERE p.name CONTAINS $value OR p.title CONTAINS $value
RETURN coalesce(p.name, p.title) AS result, labels(p)[0] AS type
LIMIT 1
"""


def map_to_database(values):
    result = ""
    for entity in values.names:
        response = graph.query(match_query, {"value": entity})
        try:
            result += f"{entity} maps to {response[0]['result']} {response[0]['type']} in database\n"
        except IndexError:
            pass
    return result


map_to_database(entities["function"])

```


```
'Casino maps to Casino Movie in database\n'

```


Custom Cypher generating chain[​](#custom-cypher-generating-chain "Direct link to Custom Cypher generating chain")
------------------------------------------------------------------------------------------------------------------

We need to define a custom Cypher prompt that takes the entity mapping information along with the schema and the user question to construct a Cypher statement. We will be using the LangChain expression language to accomplish that.

```
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Generate Cypher statement based on natural language input
cypher_template = """Based on the Neo4j graph schema below, write a Cypher query that would answer the user's question:
{schema}
Entities in the question map to the following database values:
{entities_list}
Question: {question}
Cypher query:"""  # noqa: E501

cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question, convert it to a Cypher query. No pre-amble.",
        ),
        ("human", cypher_template),
    ]
)

cypher_response = (
    RunnablePassthrough.assign(names=entity_chain)
    | RunnablePassthrough.assign(
        entities_list=lambda x: map_to_database(x["names"]["function"]),
        schema=lambda _: graph.get_schema,
    )
    | cypher_prompt
    | llm.bind(stop=["\nCypherResult:"])
    | StrOutputParser()
)

```


```
cypher = cypher_response.invoke({"question": "Who played in Casino movie?"})
cypher

```


```
'MATCH (:Movie {title: "Casino"})<-[:ACTED_IN]-(actor)\nRETURN actor.name'

```


Generating answers based on database results[​](#generating-answers-based-on-database-results "Direct link to Generating answers based on database results")
------------------------------------------------------------------------------------------------------------------------------------------------------------

Now that we have a chain that generates the Cypher statement, we need to execute the Cypher statement against the database and send the database results back to an LLM to generate the final answer. Again, we will be using LCEL.

```
from langchain.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema

# Cypher validation tool for relationship directions
corrector_schema = [
    Schema(el["start"], el["type"], el["end"])
    for el in graph.structured_schema.get("relationships")
]
cypher_validation = CypherQueryCorrector(corrector_schema)

# Generate natural language response based on database results
response_template = """Based on the the question, Cypher query, and Cypher response, write a natural language response:
Question: {question}
Cypher query: {query}
Cypher Response: {response}"""  # noqa: E501

response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question and Cypher response, convert it to a natural"
            " language answer. No pre-amble.",
        ),
        ("human", response_template),
    ]
)

chain = (
    RunnablePassthrough.assign(query=cypher_response)
    | RunnablePassthrough.assign(
        response=lambda x: graph.query(cypher_validation(x["query"])),
    )
    | response_prompt
    | llm
    | StrOutputParser()
)

```


```
chain.invoke({"question": "Who played in Casino movie?"})

```


```
'Joe Pesci, Robert De Niro, Sharon Stone, and James Woods played in the movie "Casino".'

```
