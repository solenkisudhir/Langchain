# CSV | 🦜️🔗 LangChain
LLMs are great for building question-answering systems over various types of data sources. In this section we’ll go over how to build Q&A systems over data stored in a CSV file(s). Like working with SQL databases, the key to working with CSV files is to give an LLM access to tools for querying and interacting with the data. The two main ways to do this are to either:

*   **RECOMMENDED**: Load the CSV(s) into a SQL database, and use the approaches outlined in the [SQL use case docs](https://python.langchain.com/docs/use_cases/sql/).
*   Give the LLM access to a Python environment where it can use libraries like Pandas to interact with the data.

⚠️ Security note ⚠️[​](#security-note "Direct link to ⚠️ Security note ⚠️")
---------------------------------------------------------------------------

Both approaches mentioned above carry significant risks. Using SQL requires executing model-generated SQL queries. Using a library like Pandas requires letting the model execute Python code. Since it is easier to tightly scope SQL connection permissions and sanitize SQL queries than it is to sandbox Python environments, **we HIGHLY recommend interacting with CSV data via SQL.** For more on general security best practices, [see here](https://python.langchain.com/docs/security/).

Setup[​](#setup "Direct link to Setup")
---------------------------------------

Dependencies for this guide:

```
%pip install -qU langchain langchain-openai langchain-community langchain-experimental pandas

```


Set required environment variables:

```
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

# Using LangSmith is recommended but not required. Uncomment below lines to use.
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

```


Download the [Titanic dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset) if you don’t already have it:

```
!wget https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv -O titanic.csv

```


```
import pandas as pd

df = pd.read_csv("titanic.csv")
print(df.shape)
print(df.columns.tolist())

```


```
(887, 8)
['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']

```


SQL[​](#sql "Direct link to SQL")
---------------------------------

Using SQL to interact with CSV data is the recommended approach because it is easier to limit permissions and sanitize queries than with arbitrary Python.

Most SQL databases make it easy to load a CSV file in as a table ([DuckDB](https://duckdb.org/docs/data/csv/overview.html), [SQLite](https://www.sqlite.org/csv.html), etc.). Once you’ve done this you can use all of the chain and agent-creating techniques outlined in the [SQL use case guide](https://python.langchain.com/docs/use_cases/sql/). Here’s a quick example of how we might do this with SQLite:

```
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

engine = create_engine("sqlite:///titanic.db")
df.to_sql("titanic", engine, index=False)

```


```
db = SQLDatabase(engine=engine)
print(db.dialect)
print(db.get_usable_table_names())
db.run("SELECT * FROM titanic WHERE Age < 2;")

```


```
"[(1, 2, 'Master. Alden Gates Caldwell', 'male', 0.83, 0, 2, 29.0), (0, 3, 'Master. Eino Viljami Panula', 'male', 1.0, 4, 1, 39.6875), (1, 3, 'Miss. Eleanor Ileen Johnson', 'female', 1.0, 1, 1, 11.1333), (1, 2, 'Master. Richard F Becker', 'male', 1.0, 2, 1, 39.0), (1, 1, 'Master. Hudson Trevor Allison', 'male', 0.92, 1, 2, 151.55), (1, 3, 'Miss. Maria Nakid', 'female', 1.0, 0, 2, 15.7417), (0, 3, 'Master. Sidney Leonard Goodwin', 'male', 1.0, 5, 2, 46.9), (1, 3, 'Miss. Helene Barbara Baclini', 'female', 0.75, 2, 1, 19.2583), (1, 3, 'Miss. Eugenie Baclini', 'female', 0.75, 2, 1, 19.2583), (1, 2, 'Master. Viljo Hamalainen', 'male', 0.67, 1, 1, 14.5), (1, 3, 'Master. Bertram Vere Dean', 'male', 1.0, 1, 2, 20.575), (1, 3, 'Master. Assad Alexander Thomas', 'male', 0.42, 0, 1, 8.5167), (1, 2, 'Master. Andre Mallet', 'male', 1.0, 0, 2, 37.0042), (1, 2, 'Master. George Sibley Richards', 'male', 0.83, 1, 1, 18.75)]"

```


And create a [SQL agent](https://python.langchain.com/docs/use_cases/sql/agents/) to interact with it:

```
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

```


```
agent_executor.invoke({"input": "what's the average age of survivors"})

```


```


> Entering new AgentExecutor chain...

Invoking: `sql_db_list_tables` with `{}`


titanic
Invoking: `sql_db_schema` with `{'table_names': 'titanic'}`



CREATE TABLE titanic (
    "Survived" BIGINT, 
    "Pclass" BIGINT, 
    "Name" TEXT, 
    "Sex" TEXT, 
    "Age" FLOAT, 
    "Siblings/Spouses Aboard" BIGINT, 
    "Parents/Children Aboard" BIGINT, 
    "Fare" FLOAT
)

/*
3 rows from titanic table:
Survived    Pclass  Name    Sex Age Siblings/Spouses Aboard Parents/Children Aboard Fare
0   3   Mr. Owen Harris Braund  male    22.0    1   0   7.25
1   1   Mrs. John Bradley (Florence Briggs Thayer) Cumings  female  38.0    1   0   71.2833
1   3   Miss. Laina Heikkinen   female  26.0    0   0   7.925
*/
Invoking: `sql_db_query` with `{'query': 'SELECT AVG(Age) AS AverageAge FROM titanic WHERE Survived = 1'}`
responded: To find the average age of survivors, I will query the "titanic" table and calculate the average of the "Age" column for the rows where "Survived" is equal to 1.

Here is the SQL query:

```sql
SELECT AVG(Age) AS AverageAge
FROM titanic
WHERE Survived = 1
```

Executing this query will give us the average age of the survivors.

[(28.408391812865496,)]The average age of the survivors is approximately 28.41 years.

> Finished chain.

```


```
{'input': "what's the average age of survivors",
 'output': 'The average age of the survivors is approximately 28.41 years.'}

```


This approach easily generalizes to multiple CSVs, since we can just load each of them into our database as it’s own table. Head to the [SQL guide](https://python.langchain.com/docs/use_cases/sql/) for more.

Pandas[​](#pandas "Direct link to Pandas")
------------------------------------------

Instead of SQL we can also use data analysis libraries like pandas and the code generating abilities of LLMs to interact with CSV data. Again, **this approach is not fit for production use cases unless you have extensive safeguards in place**. For this reason, our code-execution utilities and constructors live in the `langchain-experimental` package.

### Chain[​](#chain "Direct link to Chain")

Most LLMs have been trained on enough pandas Python code that they can generate it just by being asked to:

```
ai_msg = llm.invoke(
    "I have a pandas DataFrame 'df' with columns 'Age' and 'Fare'. Write code to compute the correlation between the two columns. Return Markdown for a Python code snippet and nothing else."
)
print(ai_msg.content)

```


```
```python
correlation = df['Age'].corr(df['Fare'])
correlation
```

```


We can combine this ability with a Python-executing tool to create a simple data analysis chain. We’ll first want to load our CSV table as a dataframe, and give the tool access to this dataframe:

```
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonAstREPLTool

df = pd.read_csv("titanic.csv")
tool = PythonAstREPLTool(locals={"df": df})
tool.invoke("df['Fare'].mean()")

```


To help enforce proper use of our Python tool, we’ll using [function calling](https://python.langchain.com/docs/modules/model_io/chat/function_calling/):

```
llm_with_tools = llm.bind_tools([tool], tool_choice=tool.name)
llm_with_tools.invoke(
    "I have a dataframe 'df' and want to know the correlation between the 'Age' and 'Fare' columns"
)

```


```
AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_6TZsNaCqOcbP7lqWudosQTd6', 'function': {'arguments': '{\n  "query": "df[[\'Age\', \'Fare\']].corr()"\n}', 'name': 'python_repl_ast'}, 'type': 'function'}]})

```


We’ll add a [OpenAI tools output parser](https://python.langchain.com/docs/modules/model_io/output_parsers/types/openai_tools/) to extract the function call as a dict:

```
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser

parser = JsonOutputKeyToolsParser(tool.name, first_tool_only=True)
(llm_with_tools | parser).invoke(
    "I have a dataframe 'df' and want to know the correlation between the 'Age' and 'Fare' columns"
)

```


```
{'query': "df[['Age', 'Fare']].corr()"}

```


And combine with a prompt so that we can just specify a question without needing to specify the dataframe info every invocation:

```
system = f"""You have access to a pandas dataframe `df`. \
Here is the output of `df.head().to_markdown()`:

```
{df.head().to_markdown()}
```

Given a user question, write the Python code to answer it. \
Return ONLY the valid Python code and nothing else. \
Don't assume you have access to any libraries other than built-in Python ones and pandas."""
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
code_chain = prompt | llm_with_tools | parser
code_chain.invoke({"question": "What's the correlation between age and fare"})

```


```
{'query': "df[['Age', 'Fare']].corr()"}

```


And lastly we’ll add our Python tool so that the generated code is actually executed:

```
chain = prompt | llm_with_tools | parser | tool  # noqa
chain.invoke({"question": "What's the correlation between age and fare"})

```


And just like that we have a simple data analysis chain. We can take a peak at the intermediate steps by looking at the LangSmith trace: [https://smith.langchain.com/public/b1309290-7212-49b7-bde2-75b39a32b49a/r](https://smith.langchain.com/public/b1309290-7212-49b7-bde2-75b39a32b49a/r)

We could add an additional LLM call at the end to generate a conversational response, so that we’re not just responding with the tool output. For this we’ll want to add a chat history `MessagesPlaceholder` to our prompt:

```
from operator import itemgetter

from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

system = f"""You have access to a pandas dataframe `df`. \
Here is the output of `df.head().to_markdown()`:

```
{df.head().to_markdown()}
```

Given a user question, write the Python code to answer it. \
Don't assume you have access to any libraries other than built-in Python ones and pandas.
Respond directly to the question once you have enough information to answer it."""
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system,
        ),
        ("human", "{question}"),
        # This MessagesPlaceholder allows us to optionally append an arbitrary number of messages
        # at the end of the prompt using the 'chat_history' arg.
        MessagesPlaceholder("chat_history", optional=True),
    ]
)


def _get_chat_history(x: dict) -> list:
    """Parse the chain output up to this point into a list of chat history messages to insert in the prompt."""
    ai_msg = x["ai_msg"]
    tool_call_id = x["ai_msg"].additional_kwargs["tool_calls"][0]["id"]
    tool_msg = ToolMessage(tool_call_id=tool_call_id, content=str(x["tool_output"]))
    return [ai_msg, tool_msg]


chain = (
    RunnablePassthrough.assign(ai_msg=prompt | llm_with_tools)
    .assign(tool_output=itemgetter("ai_msg") | parser | tool)
    .assign(chat_history=_get_chat_history)
    .assign(response=prompt | llm | StrOutputParser())
    .pick(["tool_output", "response"])
)

```


```
chain.invoke({"question": "What's the correlation between age and fare"})

```


```
{'tool_output': 0.11232863699941621,
 'response': 'The correlation between age and fare is approximately 0.112.'}

```


Here’s the LangSmith trace for this run: [https://smith.langchain.com/public/ca689f8a-5655-4224-8bcf-982080744462/r](https://smith.langchain.com/public/ca689f8a-5655-4224-8bcf-982080744462/r)

### Agent[​](#agent "Direct link to Agent")

For complex questions it can be helpful for an LLM to be able to iteratively execute code while maintaining the inputs and outputs of its previous executions. This is where Agents come into play. They allow an LLM to decide how many times a tool needs to be invoked and keep track of the executions it’s made so far. The [create\_pandas\_dataframe\_agent](https://api.python.langchain.com/en/latest/agents/langchain_experimental.agents.agent_toolkits.pandas.base.create_pandas_dataframe_agent.html) is a built-in agent that makes it easy to work with dataframes:

```
from langchain_experimental.agents import create_pandas_dataframe_agent

agent = create_pandas_dataframe_agent(llm, df, agent_type="openai-tools", verbose=True)
agent.invoke(
    {
        "input": "What's the correlation between age and fare? is that greater than the correlation between fare and survival?"
    }
)

```


```


> Entering new AgentExecutor chain...

Invoking: `python_repl_ast` with `{'query': "df[['Age', 'Fare']].corr()"}`


           Age      Fare
Age   1.000000  0.112329
Fare  0.112329  1.000000
Invoking: `python_repl_ast` with `{'query': "df[['Fare', 'Survived']].corr()"}`


              Fare  Survived
Fare      1.000000  0.256179
Survived  0.256179  1.000000The correlation between age and fare is 0.112329, while the correlation between fare and survival is 0.256179. Therefore, the correlation between fare and survival is greater than the correlation between age and fare.

> Finished chain.

```


```
{'input': "What's the correlation between age and fare? is that greater than the correlation between fare and survival?",
 'output': 'The correlation between age and fare is 0.112329, while the correlation between fare and survival is 0.256179. Therefore, the correlation between fare and survival is greater than the correlation between age and fare.'}

```


Here’s the LangSmith trace for this run: [https://smith.langchain.com/public/8e6c23cc-782c-4203-bac6-2a28c770c9f0/r](https://smith.langchain.com/public/8e6c23cc-782c-4203-bac6-2a28c770c9f0/r)

### Multiple CSVs[​](#multiple-csvs "Direct link to Multiple CSVs")

To handle multiple CSVs (or dataframes) we just need to pass multiple dataframes to our Python tool. Our `create_pandas_dataframe_agent` constructor can do this out of the box, we can pass in a list of dataframes instead of just one. If we’re constructing a chain ourselves, we can do something like:

```
df_1 = df[["Age", "Fare"]]
df_2 = df[["Fare", "Survived"]]

tool = PythonAstREPLTool(locals={"df_1": df_1, "df_2": df_2})
llm_with_tool = llm.bind_tools(tools=[tool], tool_choice=tool.name)
df_template = """```python
{df_name}.head().to_markdown()
>>> {df_head}
```"""
df_context = "\n\n".join(
    df_template.format(df_head=_df.head().to_markdown(), df_name=df_name)
    for _df, df_name in [(df_1, "df_1"), (df_2, "df_2")]
)

system = f"""You have access to a number of pandas dataframes. \
Here is a sample of rows from each dataframe and the python code that was used to generate the sample:

{df_context}

Given a user question about the dataframes, write the Python code to answer it. \
Don't assume you have access to any libraries other than built-in Python ones and pandas. \
Make sure to refer only to the variables mentioned above."""
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])

chain = prompt | llm_with_tool | parser | tool
chain.invoke(
    {
        "question": "return the difference in the correlation between age and fare and the correlation between fare and survival"
    }
)

```


Here’s the LangSmith trace for this run: [https://smith.langchain.com/public/653e499f-179c-4757-8041-f5e2a5f11fcc/r](https://smith.langchain.com/public/653e499f-179c-4757-8041-f5e2a5f11fcc/r)

### Sandboxed code execution[​](#sandboxed-code-execution "Direct link to Sandboxed code execution")

There are a number of tools like [E2B](https://python.langchain.com/docs/integrations/tools/e2b_data_analysis/) and [Bearly](https://python.langchain.com/docs/integrations/tools/bearly/) that provide sandboxed environments for Python code execution, to allow for safer code-executing chains and agents.

Next steps[​](#next-steps "Direct link to Next steps")
------------------------------------------------------

For more advanced data analysis applications we recommend checking out:

*   [SQL use case](https://python.langchain.com/docs/use_cases/sql/): Many of the challenges of working with SQL db’s and CSV’s are generic to any structured data type, so it’s useful to read the SQL techniques even if you’re using Pandas for CSV data analysis.
*   [Tool use](https://python.langchain.com/docs/use_cases/tool_use/): Guides on general best practices when working with chains and agents that invoke tools
*   [Agents](https://python.langchain.com/docs/modules/agents/): Understand the fundamentals of building LLM agents.
*   Integrations: Sandboxed envs like [E2B](https://python.langchain.com/docs/integrations/tools/e2b_data_analysis/) and [Bearly](https://python.langchain.com/docs/integrations/tools/bearly/), utilities like [SQLDatabase](https://api.python.langchain.com/en/latest/utilities/langchain_community.utilities.sql_database.SQLDatabase.html#langchain_community.utilities.sql_database.SQLDatabase), related agents like [Spark DataFrame agent](https://python.langchain.com/docs/integrations/toolkits/spark/).