# Query validation | 🦜️🔗 LangChain
Perhaps the most error-prone part of any SQL chain or agent is writing valid and safe SQL queries. In this guide we’ll go over some strategies for validating our queries and handling invalid queries.

Setup[​](#setup "Direct link to Setup")
---------------------------------------

First, get required packages and set environment variables:

```
%pip install --upgrade --quiet  langchain langchain-community langchain-openai

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


The below example will use a SQLite connection with Chinook database. Follow [these installation steps](https://database.guide/2-sample-databases-sqlite/) to create `Chinook.db` in the same directory as this notebook:

*   Save [this file](https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql) as `Chinook_Sqlite.sql`
*   Run `sqlite3 Chinook.db`
*   Run `.read Chinook_Sqlite.sql`
*   Test `SELECT * FROM Artist LIMIT 10;`

Now, `Chinhook.db` is in our directory and we can interface with it using the SQLAlchemy-driven `SQLDatabase` class:

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


Query checker[​](#query-checker "Direct link to Query checker")
---------------------------------------------------------------

Perhaps the simplest strategy is to ask the model itself to check the original query for common mistakes. Suppose we have the following SQL query chain:

```
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = create_sql_query_chain(llm, db)

```


And we want to validate its outputs. We can do so by extending the chain with a second prompt and model call:

```
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

system = """Double check the user's {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

Output the final SQL query only."""
prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{query}")]
).partial(dialect=db.dialect)
validation_chain = prompt | llm | StrOutputParser()

full_chain = {"query": chain} | validation_chain

```


```
query = full_chain.invoke(
    {
        "question": "What's the average Invoice from an American customer whose Fax is missing since 2003 but before 2010"
    }
)
query

```


```
"SELECT AVG(Invoice.Total) AS AverageInvoice\nFROM Invoice\nJOIN Customer ON Invoice.CustomerId = Customer.CustomerId\nWHERE Customer.Country = 'USA'\nAND Customer.Fax IS NULL\nAND Invoice.InvoiceDate >= '2003-01-01'\nAND Invoice.InvoiceDate < '2010-01-01'"

```


The obvious downside of this approach is that we need to make two model calls instead of one to generate our query. To get around this we can try to perform the query generation and query check in a single model invocation:

```
system = """You are a {dialect} expert. Given an input question, creat a syntactically correct {dialect} query to run.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per {dialect}. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".

Only use the following tables:
{table_info}

Write an initial draft of the query. Then double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

Use format:

First draft: <<FIRST_DRAFT_QUERY>>
Final answer: <<FINAL_ANSWER_QUERY>>
"""
prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{input}")]
).partial(dialect=db.dialect)


def parse_final_answer(output: str) -> str:
    return output.split("Final answer: ")[1]


chain = create_sql_query_chain(llm, db, prompt=prompt) | parse_final_answer
prompt.pretty_print()

```


```
================================ System Message ================================

You are a {dialect} expert. Given an input question, creat a syntactically correct {dialect} query to run.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per {dialect}. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".

Only use the following tables:
{table_info}

Write an initial draft of the query. Then double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

Use format:

First draft: <<FIRST_DRAFT_QUERY>>
Final answer: <<FINAL_ANSWER_QUERY>>


================================ Human Message =================================

{input}

```


```
query = chain.invoke(
    {
        "question": "What's the average Invoice from an American customer whose Fax is missing since 2003 but before 2010"
    }
)
query

```


```
"\nSELECT AVG(i.Total) AS AverageInvoice\nFROM Invoice i\nJOIN Customer c ON i.CustomerId = c.CustomerId\nWHERE c.Country = 'USA' AND c.Fax IS NULL AND i.InvoiceDate >= date('2003-01-01') AND i.InvoiceDate < date('2010-01-01')"

```


Human-in-the-loop[​](#human-in-the-loop "Direct link to Human-in-the-loop")
---------------------------------------------------------------------------

In some cases our data is sensitive enough that we never want to execute a SQL query without a human approving it first. Head to the [Tool use: Human-in-the-loop](https://python.langchain.com/docs/use_cases/tool_use/human_in_the_loop/) page to learn how to add a human-in-the-loop to any tool, chain or agent.

Error handling[​](#error-handling "Direct link to Error handling")
------------------------------------------------------------------

At some point, the model will make a mistake and craft an invalid SQL query. Or an issue will arise with our database. Or the model API will go down. We’ll want to add some error handling behavior to our chains and agents so that we fail gracefully in these situations, and perhaps even automatically recover. To learn about error handling with tools, head to the [Tool use: Error handling](https://python.langchain.com/docs/use_cases/tool_use/tool_error_handling/) page.