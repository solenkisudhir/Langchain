# Large databases | 🦜️🔗 LangChain
In order to write valid queries against a database, we need to feed the model the table names, table schemas, and feature values for it to query over. When there are many tables, columns, and/or high-cardinality columns, it becomes impossible for us to dump the full information about our database in every prompt. Instead, we must find ways to dynamically insert into the prompt only the most relevant information. Let’s take a look at some techniques for doing this.

Setup[​](#setup "Direct link to Setup")
---------------------------------------

First, get required packages and set environment variables:

```
%pip install --upgrade --quiet  langchain langchain-community langchain-openai

```


```

[notice] A new release of pip is available: 23.2.1 -> 23.3.2
[notice] To update, run: pip install --upgrade pip
Note: you may need to restart the kernel to use updated packages.

```


We default to OpenAI models in this guide, but you can swap them out for the model provider of your choice.

```
import getpass
import os

# os.environ["OPENAI_API_KEY"] = getpass.getpass()

# Uncomment the below to use LangSmith. Not required.
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
# os.environ["LANGCHAIN_TRACING_V2"] = "true"

```


The below example will use a SQLite connection with Chinook database. Follow [these installation steps](https://database.guide/2-sample-databases-sqlite/) to create `Chinook.db` in the same directory as this notebook:

*   Save [this file](https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql) as `Chinook_Sqlite.sql`
*   Run `sqlite3 Chinook.db`
*   Run `.read Chinook_Sqlite.sql`
*   Test `SELECT * FROM Artist LIMIT 10;`

Now, `Chinhook.db` is in our directory and we can interface with it using the SQLAlchemy-driven [SQLDatabase](https://api.python.langchain.com/en/latest/utilities/langchain_community.utilities.sql_database.SQLDatabase.html) class:

```
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.dialect)
print(db.get_usable_table_names())
db.run("SELECT * FROM Artist LIMIT 10;")

```


```
sqlite
['Album', 'Artist', 'Customer', 'Employee', 'Genre', 'Invoice', 'InvoiceLine', 'MediaType', 'Playlist', 'PlaylistTrack', 'Track']

```


```
"[(1, 'AC/DC'), (2, 'Accept'), (3, 'Aerosmith'), (4, 'Alanis Morissette'), (5, 'Alice In Chains'), (6, 'Antônio Carlos Jobim'), (7, 'Apocalyptica'), (8, 'Audioslave'), (9, 'BackBeat'), (10, 'Billy Cobham')]"

```


Many tables[​](#many-tables "Direct link to Many tables")
---------------------------------------------------------

One of the main pieces of information we need to include in our prompt is the schemas of the relevant tables. When we have very many tables, we can’t fit all of the schemas in a single prompt. What we can do in such cases is first extract the names of the tables related to the user input, and then include only their schemas.

One easy and reliable way to do this is using OpenAI function-calling and Pydantic models. LangChain comes with a built-in [create\_extraction\_chain\_pydantic](https://api.python.langchain.com/en/latest/chains/langchain.chains.openai_tools.extraction.create_extraction_chain_pydantic.html) chain that lets us do just this:

```
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)


class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")


table_names = "\n".join(db.get_usable_table_names())
system = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:

{table_names}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""
table_chain = create_extraction_chain_pydantic(Table, llm, system_message=system)
table_chain.invoke({"input": "What are all the genres of Alanis Morisette songs"})

```


```
[Table(name='Genre'), Table(name='Artist'), Table(name='Track')]

```


This works pretty well! Except, as we’ll see below, we actually need a few other tables as well. This would be pretty difficult for the model to know based just on the user question. In this case, we might think to simplify our model’s job by grouping the tables together. We’ll just ask the model to choose between categories “Music” and “Business”, and then take care of selecting all the relevant tables from there:

```
system = """Return the names of the SQL tables that are relevant to the user question. \
The tables are:

Music
Business"""
category_chain = create_extraction_chain_pydantic(Table, llm, system_message=system)
category_chain.invoke({"input": "What are all the genres of Alanis Morisette songs"})

```


```
from typing import List


def get_tables(categories: List[Table]) -> List[str]:
    tables = []
    for category in categories:
        if category.name == "Music":
            tables.extend(
                [
                    "Album",
                    "Artist",
                    "Genre",
                    "MediaType",
                    "Playlist",
                    "PlaylistTrack",
                    "Track",
                ]
            )
        elif category.name == "Business":
            tables.extend(["Customer", "Employee", "Invoice", "InvoiceLine"])
    return tables


table_chain = category_chain | get_tables  # noqa
table_chain.invoke({"input": "What are all the genres of Alanis Morisette songs"})

```


```
['Album', 'Artist', 'Genre', 'MediaType', 'Playlist', 'PlaylistTrack', 'Track']

```


Now that we’ve got a chain that can output the relevant tables for any query we can combine this with our [create\_sql\_query\_chain](https://api.python.langchain.com/en/latest/chains/langchain.chains.sql_database.query.create_sql_query_chain.html), which can accept a list of `table_names_to_use` to determine which table schemas are included in the prompt:

```
from operator import itemgetter

from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough

query_chain = create_sql_query_chain(llm, db)
# Convert "question" key to the "input" key expected by current table_chain.
table_chain = {"input": itemgetter("question")} | table_chain
# Set table_names_to_use using table_chain.
full_chain = RunnablePassthrough.assign(table_names_to_use=table_chain) | query_chain

```


```
query = full_chain.invoke(
    {"question": "What are all the genres of Alanis Morisette songs"}
)
print(query)

```


```
SELECT "Genre"."Name"
FROM "Genre"
JOIN "Track" ON "Genre"."GenreId" = "Track"."GenreId"
JOIN "Album" ON "Track"."AlbumId" = "Album"."AlbumId"
JOIN "Artist" ON "Album"."ArtistId" = "Artist"."ArtistId"
WHERE "Artist"."Name" = 'Alanis Morissette'

```


```
"[('Rock',), ('Rock',), ('Rock',), ('Rock',), ('Rock',), ('Rock',), ('Rock',), ('Rock',), ('Rock',), ('Rock',), ('Rock',), ('Rock',), ('Rock',)]"

```


We might rephrase our question slightly to remove redundancy in the answer

```
query = full_chain.invoke(
    {"question": "What is the set of all unique genres of Alanis Morisette songs"}
)
print(query)

```


```
SELECT DISTINCT g.Name
FROM Genre g
JOIN Track t ON g.GenreId = t.GenreId
JOIN Album a ON t.AlbumId = a.AlbumId
JOIN Artist ar ON a.ArtistId = ar.ArtistId
WHERE ar.Name = 'Alanis Morissette'

```


We can see the [LangSmith trace](https://smith.langchain.com/public/20b8ef90-1dac-4754-90f0-6bc11203c50a/r) for this run here.

We’ve seen how to dynamically include a subset of table schemas in a prompt within a chain. Another possible approach to this problem is to let an Agent decide for itself when to look up tables by giving it a Tool to do so. You can see an example of this in the [SQL: Agents](https://python.langchain.com/docs/use_cases/sql/agents/) guide.

High-cardinality columns[​](#high-cardinality-columns "Direct link to High-cardinality columns")
------------------------------------------------------------------------------------------------

In order to filter columns that contain proper nouns such as addresses, song names or artists, we first need to double-check the spelling in order to filter the data correctly.

One naive strategy it to create a vector store with all the distinct proper nouns that exist in the database. We can then query that vector store each user input and inject the most relevant proper nouns into the prompt.

First we need the unique values for each entity we want, for which we define a function that parses the result into a list of elements:

```
import ast
import re


def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return res


proper_nouns = query_as_list(db, "SELECT Name FROM Artist")
proper_nouns += query_as_list(db, "SELECT Title FROM Album")
proper_nouns += query_as_list(db, "SELECT Name FROM Genre")
len(proper_nouns)
proper_nouns[:5]

```


```
['AC/DC', 'Accept', 'Aerosmith', 'Alanis Morissette', 'Alice In Chains']

```


Now we can embed and store all of our values in a vector database:

```
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

vector_db = FAISS.from_texts(proper_nouns, OpenAIEmbeddings())
retriever = vector_db.as_retriever(search_kwargs={"k": 15})

```


And put together a query construction chain that first retrieves values from the database and inserts them into the prompt:

```
from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

system = """You are a SQLite expert. Given an input question, create a syntactically \
correct SQLite query to run. Unless otherwise specificed, do not return more than \
{top_k} rows.\n\nHere is the relevant table info: {table_info}\n\nHere is a non-exhaustive \
list of possible feature values. If filtering on a feature value make sure to check its spelling \
against this list first:\n\n{proper_nouns}"""

prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{input}")])

query_chain = create_sql_query_chain(llm, db, prompt=prompt)
retriever_chain = (
    itemgetter("question")
    | retriever
    | (lambda docs: "\n".join(doc.page_content for doc in docs))
)
chain = RunnablePassthrough.assign(proper_nouns=retriever_chain) | query_chain

```


To try out our chain, let’s see what happens when we try filtering on “elenis moriset”, a mispelling of Alanis Morissette, without and with retrieval:

```
# Without retrieval
query = query_chain.invoke(
    {"question": "What are all the genres of elenis moriset songs", "proper_nouns": ""}
)
print(query)
db.run(query)

```


```
SELECT DISTINCT Genre.Name
FROM Genre
JOIN Track ON Genre.GenreId = Track.GenreId
JOIN Album ON Track.AlbumId = Album.AlbumId
JOIN Artist ON Album.ArtistId = Artist.ArtistId
WHERE Artist.Name = 'Elenis Moriset'

```


```
# With retrieval
query = chain.invoke({"question": "What are all the genres of elenis moriset songs"})
print(query)
db.run(query)

```


```
SELECT DISTINCT Genre.Name
FROM Genre
JOIN Track ON Genre.GenreId = Track.GenreId
JOIN Album ON Track.AlbumId = Album.AlbumId
JOIN Artist ON Album.ArtistId = Artist.ArtistId
WHERE Artist.Name = 'Alanis Morissette'

```


We can see that with retrieval we’re able to correct the spelling and get back a valid result.

Another possible approach to this problem is to let an Agent decide for itself when to look up proper nouns. You can see an example of this in the [SQL: Agents](https://python.langchain.com/docs/use_cases/sql/agents/) guide.