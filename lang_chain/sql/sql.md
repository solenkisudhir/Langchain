# SQL | 🦜️🔗 LangChain
One of the most common types of databases that we can build Q&A systems for are SQL databases. LangChain comes with a number of built-in chains and agents that are compatible with any SQL dialect supported by SQLAlchemy (e.g., MySQL, PostgreSQL, Oracle SQL, Databricks, SQLite). They enable use cases such as:

*   Generating queries that will be run based on natural language questions,
*   Creating chatbots that can answer questions based on database data,
*   Building custom dashboards based on insights a user wants to analyze,

and much more.

⚠️ Security note ⚠️[​](#security-note "Direct link to ⚠️ Security note ⚠️")
---------------------------------------------------------------------------

Building Q&A systems of SQL databases requires executing model-generated SQL queries. There are inherent risks in doing this. Make sure that your database connection permissions are always scoped as narrowly as possible for your chain/agent’s needs. This will mitigate though not eliminate the risks of building a model-driven system. For more on general security best practices, [see here](https://python.langchain.com/docs/security/).

![sql_usecase.png](https://python.langchain.com/assets/images/sql_usecase-d432701261f05ab69b38576093718cf3.png)

Quickstart[​](#quickstart "Direct link to Quickstart")
------------------------------------------------------

Head to the **[Quickstart](https://python.langchain.com/docs/use_cases/sql/quickstart/)** page to get started.

Advanced[​](#advanced "Direct link to Advanced")
------------------------------------------------

Once you’ve familiarized yourself with the basics, you can head to the advanced guides:

*   [Agents](https://python.langchain.com/docs/use_cases/sql/agents/): Building agents that can interact with SQL DBs.
*   [Prompting strategies](https://python.langchain.com/docs/use_cases/sql/prompting/): Strategies for improving SQL query generation.
*   [Query validation](https://python.langchain.com/docs/use_cases/sql/query_checking/): How to validate SQL queries.
*   [Large databases](https://python.langchain.com/docs/use_cases/sql/large_db/): How to interact with DBs with many tables and high-cardinality columns.