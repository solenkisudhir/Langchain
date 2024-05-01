# Add chat history | 🦜️🔗 LangChain
In many Q&A applications we want to allow the user to have a back-and-forth conversation, meaning the application needs some sort of “memory” of past questions and answers, and some logic for incorporating those into its current thinking.

In this guide we focus on **adding logic for incorporating historical messages.** Further details on chat history management is [covered here](https://python.langchain.com/docs/expression_language/how_to/message_history/).

We’ll work off of the Q&A app we built over the [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) blog post by Lilian Weng in the [Quickstart](https://python.langchain.com/docs/use_cases/question_answering/quickstart/). We’ll need to update two things about our existing app:

1.  **Prompt**: Update our prompt to support historical messages as an input.
2.  **Contextualizing questions**: Add a sub-chain that takes the latest user question and reformulates it in the context of the chat history. This is needed in case the latest question references some context from past messages. For example, if a user asks a follow-up question like “Can you elaborate on the second point?”, this cannot be understood without the context of the previous message. Therefore we can’t effectively perform retrieval with a question like this.

Setup[​](#setup "Direct link to Setup")
---------------------------------------

### Dependencies[​](#dependencies "Direct link to Dependencies")

We’ll use an OpenAI chat model and embeddings and a Chroma vector store in this walkthrough, but everything shown here works with any [ChatModel](https://python.langchain.com/docs/modules/model_io/chat/) or [LLM](https://python.langchain.com/docs/modules/model_io/llms/), [Embeddings](https://python.langchain.com/docs/modules/data_connection/text_embedding/), and [VectorStore](https://python.langchain.com/docs/modules/data_connection/vectorstores/) or [Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/).

We’ll use the following packages:

```
%pip install --upgrade --quiet  langchain langchain-community langchainhub langchain-openai langchain-chroma bs4

```


We need to set environment variable `OPENAI_API_KEY`, which can be done directly or loaded from a `.env` file like so:

```
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

# import dotenv

# dotenv.load_dotenv()

```


### LangSmith[​](#langsmith "Direct link to LangSmith")

Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls. As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent. The best way to do this is with [LangSmith](https://smith.langchain.com/).

Note that LangSmith is not needed, but it is helpful. If you do want to use LangSmith, after you sign up at the link above, make sure to set your environment variables to start logging traces:

```
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

```


Chain without chat history[​](#chain-without-chat-history "Direct link to Chain without chat history")
------------------------------------------------------------------------------------------------------

Here is the Q&A app we built over the [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) blog post by Lilian Weng in the [Quickstart](https://python.langchain.com/docs/use_cases/question_answering/quickstart/):

```
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

```


```
# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

```


```
rag_chain.invoke("What is Task Decomposition?")

```


```
'Task Decomposition is a technique used to break down complex tasks into smaller and simpler steps. This approach helps agents to plan and execute tasks more efficiently by dividing them into manageable subgoals. Task decomposition can be achieved through various methods, including using prompting techniques, task-specific instructions, or human inputs.'

```


Contextualizing the question[​](#contextualizing-the-question "Direct link to Contextualizing the question")
------------------------------------------------------------------------------------------------------------

First we’ll need to define a sub-chain that takes historical messages and the latest user question, and reformulates the question if it makes reference to any information in the historical information.

We’ll use a prompt that includes a `MessagesPlaceholder` variable under the name “chat\_history”. This allows us to pass in a list of Messages to the prompt using the “chat\_history” input key, and these messages will be inserted after the system message and before the human message containing the latest question.

Note that we leverage a helper function [create\_history\_aware\_retriever](https://api.python.langchain.com/en/latest/chains/langchain.chains.history_aware_retriever.create_history_aware_retriever.html) for this step, which manages the case where `chat_history` is empty, and otherwise applies `prompt | llm | StrOutputParser() | retriever` in sequence.

`create_history_aware_retriever` constructs a chain that accepts keys `input` and `chat_history` as input, and has the same output schema as a retriever.

```
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

```


This chain prepends a rephrasing of the input query to our retriever, so that the retrieval incorporates the context of the conversation.

Chain with chat history[​](#chain-with-chat-history "Direct link to Chain with chat history")
---------------------------------------------------------------------------------------------

And now we can build our full QA chain.

Here we use [create\_stuff\_documents\_chain](https://api.python.langchain.com/en/latest/chains/langchain.chains.combine_documents.stuff.create_stuff_documents_chain.html) to generate a `question_answer_chain`, with input keys `context`, `chat_history`, and `input`– it accepts the retrieved context alongside the conversation history and query to generate an answer.

We build our final `rag_chain` with [create\_retrieval\_chain](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval.create_retrieval_chain.html). This chain applies the `history_aware_retriever` and `question_answer_chain` in sequence, retaining intermediate outputs such as the retrieved context for convenience. It has input keys `input` and `chat_history`, and includes `input`, `chat_history`, `context`, and `answer` in its output.

```
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

```


```
from langchain_core.messages import HumanMessage

chat_history = []

question = "What is Task Decomposition?"
ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=question), ai_msg_1["answer"]])

second_question = "What are common ways of doing it?"
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})

print(ai_msg_2["answer"])

```


```
Task decomposition can be done in several common ways, including using Language Model (LLM) with simple prompting like "Steps for XYZ" or "What are the subgoals for achieving XYZ?", providing task-specific instructions tailored to the specific task at hand, or incorporating human inputs to guide the decomposition process. These methods help in breaking down complex tasks into smaller, more manageable subtasks for efficient execution.

```


### Returning sources[​](#returning-sources "Direct link to Returning sources")

Often in Q&A applications it’s important to show users the sources that were used to generate the answer. LangChain’s built-in `create_retrieval_chain` will propagate retrieved source documents through to the output in the `"context"` key:

```
for document in ai_msg_2["context"]:
    print(document)
    print()

```


```
page_content='Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.\nTask decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.' metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}

page_content='Fig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#\nChain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.' metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}

page_content='Resources:\n1. Internet access for searches and information gathering.\n2. Long Term memory management.\n3. GPT-3.5 powered Agents for delegation of simple tasks.\n4. File output.\n\nPerformance Evaluation:\n1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.\n2. Constructively self-criticize your big-picture behavior constantly.\n3. Reflect on past decisions and strategies to refine your approach.\n4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.' metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}

page_content='Fig. 11. Illustration of how HuggingGPT works. (Image source: Shen et al. 2023)\nThe system comprises of 4 stages:\n(1) Task planning: LLM works as the brain and parses the user requests into multiple tasks. There are four attributes associated with each task: task type, ID, dependencies, and arguments. They use few-shot examples to guide LLM to do task parsing and planning.\nInstruction:' metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}

```


Tying it together[​](#tying-it-together "Direct link to Tying it together")
---------------------------------------------------------------------------

![](https://python.langchain.com/assets/images/conversational_retrieval_chain-5c7a96abe29e582bc575a0a0d63f86b0.png)

Here we’ve gone over how to add application logic for incorporating historical outputs, but we’re still manually updating the chat history and inserting it into each input. In a real Q&A application we’ll want some way of persisting chat history and some way of automatically inserting and updating it.

For this we can use:

*   [BaseChatMessageHistory](https://python.langchain.com/docs/modules/memory/chat_messages/): Store chat history.
*   [RunnableWithMessageHistory](https://python.langchain.com/docs/expression_language/how_to/message_history/): Wrapper for an LCEL chain and a `BaseChatMessageHistory` that handles injecting chat history into inputs and updating it after each invocation.

For a detailed walkthrough of how to use these classes together to create a stateful conversational chain, head to the [How to add message history (memory)](https://python.langchain.com/docs/expression_language/how_to/message_history/) LCEL page.

Below, we implement a simple example of the second option, in which chat histories are stored in a simple dict.

For convenience, we tie together all of the necessary steps in a single code cell:

```
import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


### Construct retriever ###
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

```


#### API Reference:

*   [create\_history\_aware\_retriever](https://api.python.langchain.com/en/latest/chains/langchain.chains.history_aware_retriever.create_history_aware_retriever.html)
*   [create\_retrieval\_chain](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval.create_retrieval_chain.html)
*   [create\_stuff\_documents\_chain](https://api.python.langchain.com/en/latest/chains/langchain.chains.combine_documents.stuff.create_stuff_documents_chain.html)
*   [ChatMessageHistory](https://api.python.langchain.com/en/latest/chat_history/langchain_core.chat_history.ChatMessageHistory.html)
*   [WebBaseLoader](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html)
*   [BaseChatMessageHistory](https://api.python.langchain.com/en/latest/chat_history/langchain_core.chat_history.BaseChatMessageHistory.html)
*   [ChatPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html)
*   [MessagesPlaceholder](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.MessagesPlaceholder.html)
*   [RunnableWithMessageHistory](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html)
*   [ChatOpenAI](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)
*   [OpenAIEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html)
*   [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)

```
conversational_rag_chain.invoke(
    {"input": "What is Task Decomposition?"},
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
)["answer"]

```


```
'Task decomposition is a technique used to break down complex tasks into smaller and simpler steps. This approach helps agents or models handle difficult tasks by dividing them into more manageable subtasks. It can be achieved through methods like Chain of Thought (CoT) or Tree of Thoughts, which guide the model in thinking step by step or exploring multiple reasoning possibilities at each step.'

```


```
conversational_rag_chain.invoke(
    {"input": "What are common ways of doing it?"},
    config={"configurable": {"session_id": "abc123"}},
)["answer"]

```


```
'Task decomposition can be done in common ways such as using Language Model (LLM) with simple prompting, task-specific instructions, or human inputs. For example, LLM can be guided with prompts like "Steps for XYZ" to break down tasks, or specific instructions like "Write a story outline" can be given for task decomposition. Additionally, human inputs can also be utilized to decompose tasks into smaller, more manageable steps.'

```
