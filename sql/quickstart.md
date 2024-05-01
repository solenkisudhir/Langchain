# Quickstart | 🦜️🔗 LangChain
In this guide we’ll go over the basic ways to create a Q&A chain and agent over a SQL database. These systems will allow us to ask a question about the data in a SQL database and get back a natural language answer. The main difference between the two is that our agent can query the database in a loop as many time as it needs to answer the question.

⚠️ Security note ⚠️[​](#security-note "Direct link to ⚠️ Security note ⚠️")
---------------------------------------------------------------------------

Building Q&A systems of SQL databases requires executing model-generated SQL queries. There are inherent risks in doing this. Make sure that your database connection permissions are always scoped as narrowly as possible for your chain/agent’s needs. This will mitigate though not eliminate the risks of building a model-driven system. For more on general security best practices, [see here](https://python.langchain.com/docs/security/).

Architecture[​](#architecture "Direct link to Architecture")
------------------------------------------------------------

At a high-level, the steps of any SQL chain and agent are:

1.  **Convert question to SQL query**: Model converts user input to a SQL query.
2.  **Execute SQL query**: Execute the SQL query.
3.  **Answer the question**: Model responds to user input using the query results.

![sql_usecase.png](https://python.langchain.com/assets/images/sql_usecase-d432701261f05ab69b38576093718cf3.png)

Setup[​](#setup "Direct link to Setup")
---------------------------------------

First, get required packages and set environment variables:

```
%pip install --upgrade --quiet  langchain langchain-community langchain-openai

```


We will use an OpenAI model in this guide.

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


Great! We’ve got a SQL database that we can query. Now let’s try hooking it up to an LLM.

Chain[​](#chain "Direct link to Chain")
---------------------------------------

Let’s create a simple chain that takes a question, turns it into a SQL query, executes the query, and uses the result to answer the original question.

### Convert question to SQL query[​](#convert-question-to-sql-query "Direct link to Convert question to SQL query")

The first step in a SQL chain or agent is to take the user input and convert it to a SQL query. LangChain comes with a built-in chain for this: [create\_sql\_query\_chain](https://api.python.langchain.com/en/latest/chains/langchain.chains.sql_database.query.create_sql_query_chain.html).

```
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = create_sql_query_chain(llm, db)
response = chain.invoke({"question": "How many employees are there"})
response

```


```
'SELECT COUNT(*) FROM Employee'

```


We can execute the query to make sure it’s valid:

We can look at the [LangSmith trace](https://smith.langchain.com/public/c8fa52ea-be46-4829-bde2-52894970b830/r) to get a better understanding of what this chain is doing. We can also inspect the chain directly for its prompts. Looking at the prompt (below), we can see that it is:

*   Dialect-specific. In this case it references SQLite explicitly.
*   Has definitions for all the available tables.
*   Has three examples rows for each table.

This technique is inspired by papers like [this](https://arxiv.org/pdf/2204.00498.pdf), which suggest showing examples rows and being explicit about tables improves performance. We can also inspect the full prompt like so:

```
chain.get_prompts()[0].pretty_print()

```


```
You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the LIMIT clause as per SQLite. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use the following tables:
{table_info}

Question: {input}

```


### Execute SQL query[​](#execute-sql-query "Direct link to Execute SQL query")

Now that we’ve generated a SQL query, we’ll want to execute it. **This is the most dangerous part of creating a SQL chain.** Consider carefully if it is OK to run automated queries over your data. Minimize the database connection permissions as much as possible. Consider adding a human approval step to you chains before query execution (see below).

We can use the `QuerySQLDatabaseTool` to easily add query execution to our chain:

```
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)
chain = write_query | execute_query
chain.invoke({"question": "How many employees are there"})

```


### Answer the question[​](#answer-the-question "Direct link to Answer the question")

Now that we’ve got a way to automatically generate and execute queries, we just need to combine the original question and SQL query result to generate a final answer. We can do this by passing question and result to the LLM once more:

```
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

answer = answer_prompt | llm | StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)

chain.invoke({"question": "How many employees are there"})

```


### Next steps[​](#next-steps "Direct link to Next steps")

For more complex query-generation, we may want to create few-shot prompts or add query-checking steps. For advanced techniques like this and more check out:

*   [Prompting strategies](https://python.langchain.com/docs/use_cases/sql/prompting/): Advanced prompt engineering techniques.
*   [Query checking](https://python.langchain.com/docs/use_cases/sql/query_checking/): Add query validation and error handling.
*   [Large databses](https://python.langchain.com/docs/use_cases/sql/large_db/): Techniques for working with large databases.

Agents[​](#agents "Direct link to Agents")
------------------------------------------

LangChain has an SQL Agent which provides a more flexible way of interacting with SQL databases. The main advantages of using the SQL Agent are:

*   It can answer questions based on the databases’ schema as well as on the databases’ content (like describing a specific table).
*   It can recover from errors by running a generated query, catching the traceback and regenerating it correctly.
*   It can answer questions that require multiple dependent queries.
*   It will save tokens by only considering the schema from relevant tables.

To initialize the agent, we use `create_sql_agent` function. This agent contains the `SQLDatabaseToolkit` which contains tools to:

*   Create and execute queries
*   Check query syntax
*   Retrieve table descriptions
*   … and more

### Initializing agent[​](#initializing-agent "Direct link to Initializing agent")

```
from langchain_community.agent_toolkits import create_sql_agent

agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

```


```
agent_executor.invoke(
    {
        "input": "List the total sales per country. Which country's customers spent the most?"
    }
)

```


```


> Entering new AgentExecutor chain...

Invoking: `sql_db_list_tables` with `{}`


Album, Artist, Customer, Employee, Genre, Invoice, InvoiceLine, MediaType, Playlist, PlaylistTrack, Track
Invoking: `sql_db_schema` with `Invoice,Customer`



CREATE TABLE "Customer" (
    "CustomerId" INTEGER NOT NULL, 
    "FirstName" NVARCHAR(40) NOT NULL, 
    "LastName" NVARCHAR(20) NOT NULL, 
    "Company" NVARCHAR(80), 
    "Address" NVARCHAR(70), 
    "City" NVARCHAR(40), 
    "State" NVARCHAR(40), 
    "Country" NVARCHAR(40), 
    "PostalCode" NVARCHAR(10), 
    "Phone" NVARCHAR(24), 
    "Fax" NVARCHAR(24), 
    "Email" NVARCHAR(60) NOT NULL, 
    "SupportRepId" INTEGER, 
    PRIMARY KEY ("CustomerId"), 
    FOREIGN KEY("SupportRepId") REFERENCES "Employee" ("EmployeeId")
)

/*
3 rows from Customer table:
CustomerId  FirstName   LastName    Company Address City    State   Country PostalCode  Phone   Fax Email   SupportRepId
1   Luís    Gonçalves   Embraer - Empresa Brasileira de Aeronáutica S.A.    Av. Brigadeiro Faria Lima, 2170 São José dos Campos SP  Brazil  12227-000   +55 (12) 3923-5555  +55 (12) 3923-5566  luisg@embraer.com.br    3
2   Leonie  Köhler  None    Theodor-Heuss-Straße 34 Stuttgart   None    Germany 70174   +49 0711 2842222    None    leonekohler@surfeu.de   5
3   François    Tremblay    None    1498 rue Bélanger   Montréal    QC  Canada  H2G 1A7 +1 (514) 721-4711   None    ftremblay@gmail.com 3
*/


CREATE TABLE "Invoice" (
    "InvoiceId" INTEGER NOT NULL, 
    "CustomerId" INTEGER NOT NULL, 
    "InvoiceDate" DATETIME NOT NULL, 
    "BillingAddress" NVARCHAR(70), 
    "BillingCity" NVARCHAR(40), 
    "BillingState" NVARCHAR(40), 
    "BillingCountry" NVARCHAR(40), 
    "BillingPostalCode" NVARCHAR(10), 
    "Total" NUMERIC(10, 2) NOT NULL, 
    PRIMARY KEY ("InvoiceId"), 
    FOREIGN KEY("CustomerId") REFERENCES "Customer" ("CustomerId")
)

/*
3 rows from Invoice table:
InvoiceId   CustomerId  InvoiceDate BillingAddress  BillingCity BillingState    BillingCountry  BillingPostalCode   Total
1   2   2009-01-01 00:00:00 Theodor-Heuss-Straße 34 Stuttgart   None    Germany 70174   1.98
2   4   2009-01-02 00:00:00 Ullevålsveien 14    Oslo    None    Norway  0171    3.96
3   8   2009-01-03 00:00:00 Grétrystraat 63 Brussels    None    Belgium 1000    5.94
*/
Invoking: `sql_db_query` with `SELECT c.Country, SUM(i.Total) AS TotalSales FROM Invoice i JOIN Customer c ON i.CustomerId = c.CustomerId GROUP BY c.Country ORDER BY TotalSales DESC LIMIT 10;`
responded: To list the total sales per country, I can query the "Invoice" and "Customer" tables. I will join these tables on the "CustomerId" column and group the results by the "BillingCountry" column. Then, I will calculate the sum of the "Total" column to get the total sales per country. Finally, I will order the results in descending order of the total sales.

Here is the SQL query:

```sql
SELECT c.Country, SUM(i.Total) AS TotalSales
FROM Invoice i
JOIN Customer c ON i.CustomerId = c.CustomerId
GROUP BY c.Country
ORDER BY TotalSales DESC
LIMIT 10;
```

Now, I will execute this query to get the total sales per country.

[('USA', 523.0600000000003), ('Canada', 303.9599999999999), ('France', 195.09999999999994), ('Brazil', 190.09999999999997), ('Germany', 156.48), ('United Kingdom', 112.85999999999999), ('Czech Republic', 90.24000000000001), ('Portugal', 77.23999999999998), ('India', 75.25999999999999), ('Chile', 46.62)]The total sales per country are as follows:

1. USA: $523.06
2. Canada: $303.96
3. France: $195.10
4. Brazil: $190.10
5. Germany: $156.48
6. United Kingdom: $112.86
7. Czech Republic: $90.24
8. Portugal: $77.24
9. India: $75.26
10. Chile: $46.62

To answer the second question, the country whose customers spent the most is the USA, with a total sales of $523.06.

> Finished chain.

```


```
{'input': "List the total sales per country. Which country's customers spent the most?",
 'output': 'The total sales per country are as follows:\n\n1. USA: $523.06\n2. Canada: $303.96\n3. France: $195.10\n4. Brazil: $190.10\n5. Germany: $156.48\n6. United Kingdom: $112.86\n7. Czech Republic: $90.24\n8. Portugal: $77.24\n9. India: $75.26\n10. Chile: $46.62\n\nTo answer the second question, the country whose customers spent the most is the USA, with a total sales of $523.06.'}

```


```
agent_executor.invoke({"input": "Describe the playlisttrack table"})

```


```


> Entering new AgentExecutor chain...

Invoking: `sql_db_list_tables` with `{}`


Album, Artist, Customer, Employee, Genre, Invoice, InvoiceLine, MediaType, Playlist, PlaylistTrack, Track
Invoking: `sql_db_schema` with `PlaylistTrack`



CREATE TABLE "PlaylistTrack" (
    "PlaylistId" INTEGER NOT NULL, 
    "TrackId" INTEGER NOT NULL, 
    PRIMARY KEY ("PlaylistId", "TrackId"), 
    FOREIGN KEY("TrackId") REFERENCES "Track" ("TrackId"), 
    FOREIGN KEY("PlaylistId") REFERENCES "Playlist" ("PlaylistId")
)

/*
3 rows from PlaylistTrack table:
PlaylistId  TrackId
1   3402
1   3389
1   3390
*/The `PlaylistTrack` table has two columns: `PlaylistId` and `TrackId`. It is a junction table that represents the many-to-many relationship between playlists and tracks. 

Here is the schema of the `PlaylistTrack` table:

```
CREATE TABLE "PlaylistTrack" (
    "PlaylistId" INTEGER NOT NULL, 
    "TrackId" INTEGER NOT NULL, 
    PRIMARY KEY ("PlaylistId", "TrackId"), 
    FOREIGN KEY("TrackId") REFERENCES "Track" ("TrackId"), 
    FOREIGN KEY("PlaylistId") REFERENCES "Playlist" ("PlaylistId")
)
```

The `PlaylistId` column is a foreign key referencing the `PlaylistId` column in the `Playlist` table. The `TrackId` column is a foreign key referencing the `TrackId` column in the `Track` table.

Here are three sample rows from the `PlaylistTrack` table:

```
PlaylistId   TrackId
1            3402
1            3389
1            3390
```

Please let me know if there is anything else I can help with.

> Finished chain.

```


```
{'input': 'Describe the playlisttrack table',
 'output': 'The `PlaylistTrack` table has two columns: `PlaylistId` and `TrackId`. It is a junction table that represents the many-to-many relationship between playlists and tracks. \n\nHere is the schema of the `PlaylistTrack` table:\n\n```\nCREATE TABLE "PlaylistTrack" (\n\t"PlaylistId" INTEGER NOT NULL, \n\t"TrackId" INTEGER NOT NULL, \n\tPRIMARY KEY ("PlaylistId", "TrackId"), \n\tFOREIGN KEY("TrackId") REFERENCES "Track" ("TrackId"), \n\tFOREIGN KEY("PlaylistId") REFERENCES "Playlist" ("PlaylistId")\n)\n```\n\nThe `PlaylistId` column is a foreign key referencing the `PlaylistId` column in the `Playlist` table. The `TrackId` column is a foreign key referencing the `TrackId` column in the `Track` table.\n\nHere are three sample rows from the `PlaylistTrack` table:\n\n```\nPlaylistId   TrackId\n1            3402\n1            3389\n1            3390\n```\n\nPlease let me know if there is anything else I can help with.'}

```


### Next steps[​](#next-steps-1 "Direct link to Next steps")

For more on how to use and customize agents head to the [Agents](https://python.langchain.com/docs/use_cases/sql/agents/) page.