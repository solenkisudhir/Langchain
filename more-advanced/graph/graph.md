# Graphs | 🦜️🔗 LangChain
One of the common types of databases that we can build Q&A systems for are graph databases. LangChain comes with a number of built-in chains and agents that are compatible with graph query language dialects like Cypher, SparQL, and others (e.g., Neo4j, MemGraph, Amazon Neptune, Kùzu, OntoText, Tigergraph). They enable use cases such as:

*   Generating queries that will be run based on natural language questions,
*   Creating chatbots that can answer questions based on database data,
*   Building custom dashboards based on insights a user wants to analyze,

and much more.

⚠️ Security note ⚠️[​](#security-note "Direct link to ⚠️ Security note ⚠️")
---------------------------------------------------------------------------

Building Q&A systems of graph databases might require executing model-generated database queries. There are inherent risks in doing this. Make sure that your database connection permissions are always scoped as narrowly as possible for your chain/agent’s needs. This will mitigate though not eliminate the risks of building a model-driven system. For more on general security best practices, [see here](https://python.langchain.com/docs/security/).

![graphgrag_usecase.png](https://python.langchain.com/assets/images/graph_usecase-34d891523e6284bb6230b38c5f8392e5.png)

> Employing database query templates within a semantic layer provides the advantage of bypassing the need for database query generation. This approach effectively eradicates security vulnerabilities linked to the generation of database queries.

Quickstart[​](#quickstart "Direct link to Quickstart")
------------------------------------------------------

Head to the **[Quickstart](https://python.langchain.com/docs/use_cases/graph/quickstart/)** page to get started.

Advanced[​](#advanced "Direct link to Advanced")
------------------------------------------------

Once you’ve familiarized yourself with the basics, you can head to the advanced guides:

*   [Prompting strategies](https://python.langchain.com/docs/use_cases/graph/prompting/): Advanced prompt engineering techniques.
*   [Mapping values](https://python.langchain.com/docs/use_cases/graph/mapping/): Techniques for mapping values from questions to database.
*   [Semantic layer](https://python.langchain.com/docs/use_cases/graph/semantic/): Techniques for implementing semantic layers.
*   [Constructing graphs](https://python.langchain.com/docs/use_cases/graph/constructing/): Techniques for constructing knowledge graphs.