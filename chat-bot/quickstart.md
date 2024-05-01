# Quickstart | 🦜️🔗 LangChain
[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/use_cases/chatbots.ipynb)

Overview[​](#overview "Direct link to Overview")
------------------------------------------------

We’ll go over an example of how to design and implement an LLM-powered chatbot. Here are a few of the high-level components we’ll be working with:

*   `Chat Models`. The chatbot interface is based around messages rather than raw text, and therefore is best suited to Chat Models rather than text LLMs. See [here](https://python.langchain.com/docs/integrations/chat/) for a list of chat model integrations and [here](https://python.langchain.com/docs/modules/model_io/chat/) for documentation on the chat model interface in LangChain. You can use `LLMs` (see [here](https://python.langchain.com/docs/modules/model_io/llms/)) for chatbots as well, but chat models have a more conversational tone and natively support a message interface.
*   `Prompt Templates`, which simplify the process of assembling prompts that combine default messages, user input, chat history, and (optionally) additional retrieved context.
*   `Chat History`, which allows a chatbot to “remember” past interactions and take them into account when responding to followup questions. [See here](https://python.langchain.com/docs/modules/memory/chat_messages/) for more information.
*   `Retrievers` (optional), which are useful if you want to build a chatbot that can use domain-specific, up-to-date knowledge as context to augment its responses. [See here](https://python.langchain.com/docs/modules/data_connection/retrievers/) for in-depth documentation on retrieval systems.

We’ll cover how to fit the above components together to create a powerful conversational chatbot.

Quickstart[​](#quickstart "Direct link to Quickstart")
------------------------------------------------------

To start, let’s install some dependencies and set the required credentials:

```
%pip install --upgrade --quiet langchain langchain-openai langchain-chroma

# Set env var OPENAI_API_KEY or load from a .env file:
import dotenv

dotenv.load_dotenv()

```


```
WARNING: You are using pip version 22.0.4; however, version 23.3.2 is available.
You should consider upgrading via the '/Users/jacoblee/.pyenv/versions/3.10.5/bin/python -m pip install --upgrade pip' command.
Note: you may need to restart the kernel to use updated packages.

```


Let’s initialize the chat model which will serve as the chatbot’s brain:

```
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

```


If we invoke our chat model, the output is an `AIMessage`:

```
from langchain_core.messages import HumanMessage

chat.invoke(
    [
        HumanMessage(
            content="Translate this sentence from English to French: I love programming."
        )
    ]
)

```


```
AIMessage(content="J'adore programmer.")

```


The model on its own does not have any concept of state. For example, if you ask a followup question:

```
chat.invoke([HumanMessage(content="What did you just say?")])

```


```
AIMessage(content='I said, "What did you just say?"')

```


We can see that it doesn’t take the previous conversation turn into context, and cannot answer the question.

To get around this, we need to pass the entire conversation history into the model. Let’s see what happens when we do that:

```
from langchain_core.messages import AIMessage

chat.invoke(
    [
        HumanMessage(
            content="Translate this sentence from English to French: I love programming."
        ),
        AIMessage(content="J'adore la programmation."),
        HumanMessage(content="What did you just say?"),
    ]
)

```


```
AIMessage(content='I said "J\'adore la programmation," which means "I love programming" in French.')

```


And now we can see that we get a good response!

This is the basic idea underpinning a chatbot’s ability to interact conversationally.

Prompt templates[​](#prompt-templates "Direct link to Prompt templates")
------------------------------------------------------------------------

Let’s define a prompt template to make formatting a bit easier. We can create a chain by piping it into the model:

```
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | chat

```


The `MessagesPlaceholder` above inserts chat messages passed into the chain’s input as `chat_history` directly into the prompt. Then, we can invoke the chain like this:

```
chain.invoke(
    {
        "messages": [
            HumanMessage(
                content="Translate this sentence from English to French: I love programming."
            ),
            AIMessage(content="J'adore la programmation."),
            HumanMessage(content="What did you just say?"),
        ],
    }
)

```


```
AIMessage(content='I said "J\'adore la programmation," which means "I love programming" in French.')

```


Message history[​](#message-history "Direct link to Message history")
---------------------------------------------------------------------

As a shortcut for managing the chat history, we can use a [`MessageHistory`](https://python.langchain.com/docs/modules/memory/chat_messages/) class, which is responsible for saving and loading chat messages. There are many built-in message history integrations that persist messages to a variety of databases, but for this quickstart we’ll use a in-memory, demo message history called `ChatMessageHistory`.

Here’s an example of using it directly:

```
from langchain.memory import ChatMessageHistory

demo_ephemeral_chat_history = ChatMessageHistory()

demo_ephemeral_chat_history.add_user_message("hi!")

demo_ephemeral_chat_history.add_ai_message("whats up?")

demo_ephemeral_chat_history.messages

```


```
[HumanMessage(content='hi!'), AIMessage(content='whats up?')]

```


Once we do that, we can pass the stored messages directly into our chain as a parameter:

```
demo_ephemeral_chat_history.add_user_message(
    "Translate this sentence from English to French: I love programming."
)

response = chain.invoke({"messages": demo_ephemeral_chat_history.messages})

response

```


```
AIMessage(content='The translation of "I love programming" in French is "J\'adore la programmation."')

```


```
demo_ephemeral_chat_history.add_ai_message(response)

demo_ephemeral_chat_history.add_user_message("What did you just say?")

chain.invoke({"messages": demo_ephemeral_chat_history.messages})

```


```
AIMessage(content='I said "J\'adore la programmation," which is the French translation for "I love programming."')

```


And now we have a basic chatbot!

While this chain can serve as a useful chatbot on its own with just the model’s internal knowledge, it’s often useful to introduce some form of `retrieval-augmented generation`, or RAG for short, over domain-specific knowledge to make our chatbot more focused. We’ll cover this next.

Retrievers[​](#retrievers "Direct link to Retrievers")
------------------------------------------------------

We can set up and use a [`Retriever`](https://python.langchain.com/docs/modules/data_connection/retrievers/) to pull domain-specific knowledge for our chatbot. To show this, let’s expand the simple chatbot we created above to be able to answer questions about LangSmith.

We’ll use [the LangSmith documentation](https://docs.smith.langchain.com/overview) as source material and store it in a vectorstore for later retrieval. Note that this example will gloss over some of the specifics around parsing and storing a data source - you can see more [in-depth documentation on creating retrieval systems here](https://python.langchain.com/docs/use_cases/question_answering/).

Let’s set up our retriever. First, we’ll install some required deps:

```
%pip install --upgrade --quiet langchain-chroma beautifulsoup4

```


```
WARNING: You are using pip version 22.0.4; however, version 23.3.2 is available.
You should consider upgrading via the '/Users/jacoblee/.pyenv/versions/3.10.5/bin/python -m pip install --upgrade pip' command.
Note: you may need to restart the kernel to use updated packages.

```


Next, we’ll use a document loader to pull data from a webpage:

```
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
data = loader.load()

```


Next, we split it into smaller chunks that the LLM’s context window can handle and store it in a vector database:

```
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

```


Then we embed and store those chunks in a vector database:

```
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

```


And finally, let’s create a retriever from our initialized vectorstore:

```
# k is the number of chunks to retrieve
retriever = vectorstore.as_retriever(k=4)

docs = retriever.invoke("how can langsmith help with testing?")

docs

```


```
[Document(page_content='You can also quickly edit examples and add them to datasets to expand the surface area of your evaluation sets or to fine-tune a model for improved quality or reduced costs.Monitoring\u200bAfter all this, your app might finally ready to go in production. LangSmith can also be used to monitor your application in much the same way that you used for debugging. You can log all traces, visualize latency and token usage statistics, and troubleshoot specific issues as they arise. Each run can also be', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
 Document(page_content='inputs, and see what happens. At some point though, our application is performing\nwell and we want to be more rigorous about testing changes. We can use a dataset\nthat we’ve constructed along the way (see above). Alternatively, we could spend some\ntime constructing a small dataset by hand. For these situations, LangSmith simplifies', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
 Document(page_content='Skip to main content🦜️🛠️ LangSmith DocsPython DocsJS/TS DocsSearchGo to AppLangSmithOverviewTracingTesting & EvaluationOrganizationsHubLangSmith CookbookOverviewOn this pageLangSmith Overview and User GuideBuilding reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.Over the past two months, we at LangChain', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
 Document(page_content='have been building and using LangSmith with the goal of bridging this gap. This is our tactical user guide to outline effective ways to use LangSmith and maximize its benefits.On by default\u200bAt LangChain, all of us have LangSmith’s tracing running in the background by default. On the Python side, this is achieved by setting environment variables, which we establish whenever we launch a virtual environment or open our bash shell and leave them set. The same principle applies to most JavaScript', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'})]

```


We can see that invoking the retriever above results in some parts of the LangSmith docs that contain information about testing that our chatbot can use as context when answering questions.

### Handling documents[​](#handling-documents "Direct link to Handling documents")

Let’s modify our previous prompt to accept documents as context. We’ll use a [`create_stuff_documents_chain`](https://api.python.langchain.com/en/latest/chains/langchain.chains.combine_documents.stuff.create_stuff_documents_chain.html#langchain.chains.combine_documents.stuff.create_stuff_documents_chain) helper function to “stuff” all of the input documents into the prompt, which also conveniently handles formatting. We use the [`ChatPromptTemplate.from_messages`](https://python.langchain.com/docs/modules/model_io/prompts/quick_start/#chatprompttemplate) method to format the message input we want to pass to the model, including a [`MessagesPlaceholder`](https://python.langchain.com/docs/modules/model_io/prompts/quick_start/#messagesplaceholder) where chat history messages will be directly injected:

```
from langchain.chains.combine_documents import create_stuff_documents_chain

chat = ChatOpenAI(model="gpt-3.5-turbo-1106")

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

```


We can invoke this `document_chain` with the raw documents we retrieved above:

```
from langchain.memory import ChatMessageHistory

demo_ephemeral_chat_history = ChatMessageHistory()

demo_ephemeral_chat_history.add_user_message("how can langsmith help with testing?")

document_chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
        "context": docs,
    }
)

```


```
'LangSmith can assist with testing by providing the capability to quickly edit examples and add them to datasets. This allows for the expansion of evaluation sets or fine-tuning of a model for improved quality or reduced costs. Additionally, LangSmith simplifies the construction of small datasets by hand, providing a convenient way to rigorously test changes in the application.'

```


Awesome! We see an answer synthesized from information in the input documents.

### Creating a retrieval chain[​](#creating-a-retrieval-chain "Direct link to Creating a retrieval chain")

Next, let’s integrate our retriever into the chain. Our retriever should retrieve information relevant to the last message we pass in from the user, so we extract it and use that as input to fetch relevant docs, which we add to the current chain as `context`. We pass `context` plus the previous `messages` into our document chain to generate a final answer.

We also use the [`RunnablePassthrough.assign()`](https://python.langchain.com/docs/expression_language/primitives/assign/) method to pass intermediate steps through at each invocation. Here’s what it looks like:

```
from typing import Dict

from langchain_core.runnables import RunnablePassthrough


def parse_retriever_input(params: Dict):
    return params["messages"][-1].content


retrieval_chain = RunnablePassthrough.assign(
    context=parse_retriever_input | retriever,
).assign(
    answer=document_chain,
)

```


```
response = retrieval_chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
    }
)

response

```


```
{'messages': [HumanMessage(content='how can langsmith help with testing?')],
 'context': [Document(page_content='You can also quickly edit examples and add them to datasets to expand the surface area of your evaluation sets or to fine-tune a model for improved quality or reduced costs.Monitoring\u200bAfter all this, your app might finally ready to go in production. LangSmith can also be used to monitor your application in much the same way that you used for debugging. You can log all traces, visualize latency and token usage statistics, and troubleshoot specific issues as they arise. Each run can also be', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
  Document(page_content='inputs, and see what happens. At some point though, our application is performing\nwell and we want to be more rigorous about testing changes. We can use a dataset\nthat we’ve constructed along the way (see above). Alternatively, we could spend some\ntime constructing a small dataset by hand. For these situations, LangSmith simplifies', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
  Document(page_content='Skip to main content🦜️🛠️ LangSmith DocsPython DocsJS/TS DocsSearchGo to AppLangSmithOverviewTracingTesting & EvaluationOrganizationsHubLangSmith CookbookOverviewOn this pageLangSmith Overview and User GuideBuilding reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.Over the past two months, we at LangChain', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
  Document(page_content='have been building and using LangSmith with the goal of bridging this gap. This is our tactical user guide to outline effective ways to use LangSmith and maximize its benefits.On by default\u200bAt LangChain, all of us have LangSmith’s tracing running in the background by default. On the Python side, this is achieved by setting environment variables, which we establish whenever we launch a virtual environment or open our bash shell and leave them set. The same principle applies to most JavaScript', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'})],
 'answer': 'LangSmith can help with testing in several ways:\n\n1. Dataset Expansion: LangSmith enables quick editing of examples and adding them to datasets, which expands the surface area of evaluation sets. This allows for more comprehensive testing of models and applications.\n\n2. Fine-Tuning Models: LangSmith facilitates the fine-tuning of models for improved quality or reduced costs. This is beneficial for optimizing the performance of models during testing.\n\n3. Monitoring: LangSmith can be used to monitor applications, log traces, visualize latency and token usage statistics, and troubleshoot specific issues as they arise during testing. This monitoring helps in ensuring the reliability and performance of the application during testing phases.\n\nOverall, LangSmith helps in making testing more rigorous and comprehensive, whether by expanding datasets, fine-tuning models, or monitoring application performance.'}

```


```
demo_ephemeral_chat_history.add_ai_message(response["answer"])

demo_ephemeral_chat_history.add_user_message("tell me more about that!")

retrieval_chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
    },
)

```


```
{'messages': [HumanMessage(content='how can langsmith help with testing?'),
  AIMessage(content='LangSmith can help with testing in several ways:\n\n1. Dataset Expansion: LangSmith enables quick editing of examples and adding them to datasets, which expands the surface area of evaluation sets. This allows for more comprehensive testing of models and applications.\n\n2. Fine-Tuning Models: LangSmith facilitates the fine-tuning of models for improved quality or reduced costs. This is beneficial for optimizing the performance of models during testing.\n\n3. Monitoring: LangSmith can be used to monitor applications, log traces, visualize latency and token usage statistics, and troubleshoot specific issues as they arise during testing. This monitoring helps in ensuring the reliability and performance of the application during testing phases.\n\nOverall, LangSmith helps in making testing more rigorous and comprehensive, whether by expanding datasets, fine-tuning models, or monitoring application performance.'),
  HumanMessage(content='tell me more about that!')],
 'context': [Document(page_content='however, there is still no complete substitute for human review to get the utmost quality and reliability from your application.', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
  Document(page_content='You can also quickly edit examples and add them to datasets to expand the surface area of your evaluation sets or to fine-tune a model for improved quality or reduced costs.Monitoring\u200bAfter all this, your app might finally ready to go in production. LangSmith can also be used to monitor your application in much the same way that you used for debugging. You can log all traces, visualize latency and token usage statistics, and troubleshoot specific issues as they arise. Each run can also be', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
  Document(page_content="against these known issues.Why is this so impactful? When building LLM applications, it’s often common to start without a dataset of any kind. This is part of the power of LLMs! They are amazing zero-shot learners, making it possible to get started as easily as possible. But this can also be a curse -- as you adjust the prompt, you're wandering blind. You don’t have any examples to benchmark your changes against.LangSmith addresses this problem by including an “Add to Dataset” button for each", metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
  Document(page_content='playground. Here, you can modify the prompt and re-run it to observe the resulting changes to the output - as many times as needed!Currently, this feature supports only OpenAI and Anthropic models and works for LLM and Chat Model calls. We plan to extend its functionality to more LLM types, chains, agents, and retrievers in the future.What is the exact sequence of events?\u200bIn complicated chains and agents, it can often be hard to understand what is going on under the hood. What calls are being', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'})],
 'answer': 'Certainly! LangSmith offers the following capabilities to aid in testing:\n\n1. Dataset Expansion: By allowing quick editing of examples and adding them to datasets, LangSmith enables the expansion of evaluation sets. This is crucial for thorough testing of models and applications, as it broadens the range of scenarios and inputs that can be used to assess performance.\n\n2. Fine-Tuning Models: LangSmith supports the fine-tuning of models to enhance their quality and reduce operational costs. This capability is valuable during testing as it enables the optimization of model performance based on specific testing requirements and objectives.\n\n3. Monitoring: LangSmith provides monitoring features that allow for the logging of traces, visualization of latency and token usage statistics, and troubleshooting of issues as they occur during testing. This real-time monitoring helps in identifying and addressing any issues that may impact the reliability and performance of the application during testing.\n\nBy leveraging these features, LangSmith enhances the testing process by enabling comprehensive dataset expansion, model fine-tuning, and real-time monitoring to ensure the quality and reliability of applications and models.'}

```


Nice! Our chatbot can now answer domain-specific questions in a conversational way.

As an aside, if you don’t want to return all the intermediate steps, you can define your retrieval chain like this using a pipe directly into the document chain instead of the final `.assign()` call:

```
retrieval_chain_with_only_answer = (
    RunnablePassthrough.assign(
        context=parse_retriever_input | retriever,
    )
    | document_chain
)

retrieval_chain_with_only_answer.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
    },
)

```


```
"LangSmith offers the capability to quickly edit examples and add them to datasets, thereby enhancing the scope of evaluation sets. This feature is particularly valuable for testing as it allows for a more thorough assessment of model performance and application behavior.\n\nFurthermore, LangSmith enables the fine-tuning of models to enhance quality and reduce costs, which can significantly impact testing outcomes. By adjusting and refining models, developers can ensure that they are thoroughly tested and optimized for various scenarios and use cases.\n\nAdditionally, LangSmith provides monitoring functionality, allowing users to log traces, visualize latency and token usage statistics, and troubleshoot specific issues as they encounter them during testing. This real-time monitoring and troubleshooting capability contribute to the overall effectiveness and reliability of the testing process.\n\nIn essence, LangSmith's features are designed to improve the quality and reliability of testing by expanding evaluation sets, fine-tuning models, and providing comprehensive monitoring capabilities. These aspects collectively contribute to a more robust and thorough testing process for applications and models."

```


Query transformation[​](#query-transformation "Direct link to Query transformation")
------------------------------------------------------------------------------------

There’s one more optimization we’ll cover here - in the above example, when we asked a followup question, `tell me more about that!`, you might notice that the retrieved docs don’t directly include information about testing. This is because we’re passing `tell me more about that!` verbatim as a query to the retriever. The output in the retrieval chain is still okay because the document chain retrieval chain can generate an answer based on the chat history, but we could be retrieving more rich and informative documents:

```
retriever.invoke("how can langsmith help with testing?")

```


```
[Document(page_content='You can also quickly edit examples and add them to datasets to expand the surface area of your evaluation sets or to fine-tune a model for improved quality or reduced costs.Monitoring\u200bAfter all this, your app might finally ready to go in production. LangSmith can also be used to monitor your application in much the same way that you used for debugging. You can log all traces, visualize latency and token usage statistics, and troubleshoot specific issues as they arise. Each run can also be', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
 Document(page_content='inputs, and see what happens. At some point though, our application is performing\nwell and we want to be more rigorous about testing changes. We can use a dataset\nthat we’ve constructed along the way (see above). Alternatively, we could spend some\ntime constructing a small dataset by hand. For these situations, LangSmith simplifies', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
 Document(page_content='Skip to main content🦜️🛠️ LangSmith DocsPython DocsJS/TS DocsSearchGo to AppLangSmithOverviewTracingTesting & EvaluationOrganizationsHubLangSmith CookbookOverviewOn this pageLangSmith Overview and User GuideBuilding reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.Over the past two months, we at LangChain', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
 Document(page_content='have been building and using LangSmith with the goal of bridging this gap. This is our tactical user guide to outline effective ways to use LangSmith and maximize its benefits.On by default\u200bAt LangChain, all of us have LangSmith’s tracing running in the background by default. On the Python side, this is achieved by setting environment variables, which we establish whenever we launch a virtual environment or open our bash shell and leave them set. The same principle applies to most JavaScript', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'})]

```


```
retriever.invoke("tell me more about that!")

```


```
[Document(page_content='however, there is still no complete substitute for human review to get the utmost quality and reliability from your application.', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
 Document(page_content='You can also quickly edit examples and add them to datasets to expand the surface area of your evaluation sets or to fine-tune a model for improved quality or reduced costs.Monitoring\u200bAfter all this, your app might finally ready to go in production. LangSmith can also be used to monitor your application in much the same way that you used for debugging. You can log all traces, visualize latency and token usage statistics, and troubleshoot specific issues as they arise. Each run can also be', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
 Document(page_content="against these known issues.Why is this so impactful? When building LLM applications, it’s often common to start without a dataset of any kind. This is part of the power of LLMs! They are amazing zero-shot learners, making it possible to get started as easily as possible. But this can also be a curse -- as you adjust the prompt, you're wandering blind. You don’t have any examples to benchmark your changes against.LangSmith addresses this problem by including an “Add to Dataset” button for each", metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
 Document(page_content='playground. Here, you can modify the prompt and re-run it to observe the resulting changes to the output - as many times as needed!Currently, this feature supports only OpenAI and Anthropic models and works for LLM and Chat Model calls. We plan to extend its functionality to more LLM types, chains, agents, and retrievers in the future.What is the exact sequence of events?\u200bIn complicated chains and agents, it can often be hard to understand what is going on under the hood. What calls are being', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'})]

```


To get around this common problem, let’s add a `query transformation` step that removes references from the input. We’ll wrap our old retriever as follows:

```
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

# We need a prompt that we can pass into an LLM to generate a transformed search query

chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

query_transform_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
        ),
    ]
)

query_transforming_retriever_chain = RunnableBranch(
    (
        lambda x: len(x.get("messages", [])) == 1,
        # If only one message, then we just pass that message's content to retriever
        (lambda x: x["messages"][-1].content) | retriever,
    ),
    # If messages, then we pass inputs to LLM chain to transform the query, then pass to retriever
    query_transform_prompt | chat | StrOutputParser() | retriever,
).with_config(run_name="chat_retriever_chain")

```


Now let’s recreate our earlier chain with this new `query_transforming_retriever_chain`. Note that this new chain accepts a dict as input and parses a string to pass to the retriever, so we don’t have to do additional parsing at the top level:

```
document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

conversational_retrieval_chain = RunnablePassthrough.assign(
    context=query_transforming_retriever_chain,
).assign(
    answer=document_chain,
)

demo_ephemeral_chat_history = ChatMessageHistory()

```


And finally, let’s invoke it!

```
demo_ephemeral_chat_history.add_user_message("how can langsmith help with testing?")

response = conversational_retrieval_chain.invoke(
    {"messages": demo_ephemeral_chat_history.messages},
)

demo_ephemeral_chat_history.add_ai_message(response["answer"])

response

```


```
{'messages': [HumanMessage(content='how can langsmith help with testing?'),
  AIMessage(content='LangSmith can assist with testing in several ways. It allows you to quickly edit examples and add them to datasets, expanding the range of evaluation sets. This can help in fine-tuning a model for improved quality or reduced costs. Additionally, LangSmith simplifies the construction of small datasets by hand, providing a convenient way to rigorously test changes in your application. Furthermore, it enables monitoring of your application by logging all traces, visualizing latency and token usage statistics, and troubleshooting specific issues as they arise.')],
 'context': [Document(page_content='You can also quickly edit examples and add them to datasets to expand the surface area of your evaluation sets or to fine-tune a model for improved quality or reduced costs.Monitoring\u200bAfter all this, your app might finally ready to go in production. LangSmith can also be used to monitor your application in much the same way that you used for debugging. You can log all traces, visualize latency and token usage statistics, and troubleshoot specific issues as they arise. Each run can also be', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
  Document(page_content='inputs, and see what happens. At some point though, our application is performing\nwell and we want to be more rigorous about testing changes. We can use a dataset\nthat we’ve constructed along the way (see above). Alternatively, we could spend some\ntime constructing a small dataset by hand. For these situations, LangSmith simplifies', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
  Document(page_content='Skip to main content🦜️🛠️ LangSmith DocsPython DocsJS/TS DocsSearchGo to AppLangSmithOverviewTracingTesting & EvaluationOrganizationsHubLangSmith CookbookOverviewOn this pageLangSmith Overview and User GuideBuilding reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.Over the past two months, we at LangChain', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
  Document(page_content='have been building and using LangSmith with the goal of bridging this gap. This is our tactical user guide to outline effective ways to use LangSmith and maximize its benefits.On by default\u200bAt LangChain, all of us have LangSmith’s tracing running in the background by default. On the Python side, this is achieved by setting environment variables, which we establish whenever we launch a virtual environment or open our bash shell and leave them set. The same principle applies to most JavaScript', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'})],
 'answer': 'LangSmith can assist with testing in several ways. It allows you to quickly edit examples and add them to datasets, expanding the range of evaluation sets. This can help in fine-tuning a model for improved quality or reduced costs. Additionally, LangSmith simplifies the construction of small datasets by hand, providing a convenient way to rigorously test changes in your application. Furthermore, it enables monitoring of your application by logging all traces, visualizing latency and token usage statistics, and troubleshooting specific issues as they arise.'}

```


```
demo_ephemeral_chat_history.add_user_message("tell me more about that!")

conversational_retrieval_chain.invoke(
    {"messages": demo_ephemeral_chat_history.messages}
)

```


```
{'messages': [HumanMessage(content='how can langsmith help with testing?'),
  AIMessage(content='LangSmith can assist with testing in several ways. It allows you to quickly edit examples and add them to datasets, expanding the range of evaluation sets. This can help in fine-tuning a model for improved quality or reduced costs. Additionally, LangSmith simplifies the construction of small datasets by hand, providing a convenient way to rigorously test changes in your application. Furthermore, it enables monitoring of your application by logging all traces, visualizing latency and token usage statistics, and troubleshooting specific issues as they arise.'),
  HumanMessage(content='tell me more about that!')],
 'context': [Document(page_content='LangSmith Overview and User Guide | 🦜️🛠️ LangSmith', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
  Document(page_content='You can also quickly edit examples and add them to datasets to expand the surface area of your evaluation sets or to fine-tune a model for improved quality or reduced costs.Monitoring\u200bAfter all this, your app might finally ready to go in production. LangSmith can also be used to monitor your application in much the same way that you used for debugging. You can log all traces, visualize latency and token usage statistics, and troubleshoot specific issues as they arise. Each run can also be', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
  Document(page_content='Skip to main content🦜️🛠️ LangSmith DocsPython DocsJS/TS DocsSearchGo to AppLangSmithOverviewTracingTesting & EvaluationOrganizationsHubLangSmith CookbookOverviewOn this pageLangSmith Overview and User GuideBuilding reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.Over the past two months, we at LangChain', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'}),
  Document(page_content='inputs, and see what happens. At some point though, our application is performing\nwell and we want to be more rigorous about testing changes. We can use a dataset\nthat we’ve constructed along the way (see above). Alternatively, we could spend some\ntime constructing a small dataset by hand. For these situations, LangSmith simplifies', metadata={'description': 'Building reliable LLM applications can be challenging. LangChain simplifies the initial setup, but there is still work needed to bring the performance of prompts, chains and agents up the level where they are reliable enough to be used in production.', 'language': 'en', 'source': 'https://docs.smith.langchain.com/overview', 'title': 'LangSmith Overview and User Guide | 🦜️🛠️ LangSmith'})],
 'answer': 'Certainly! LangSmith simplifies the process of constructing and editing datasets, which is essential for testing and fine-tuning models. By quickly editing examples and adding them to datasets, you can expand the surface area of your evaluation sets, leading to improved model quality and potentially reduced costs. Additionally, LangSmith provides monitoring capabilities for your application, allowing you to log all traces, visualize latency and token usage statistics, and troubleshoot specific issues as they arise. This comprehensive monitoring functionality helps ensure the reliability and performance of your application in production.'}

```


To help you understand what’s happening internally, [this LangSmith trace](https://smith.langchain.com/public/42f8993b-7d19-42d3-990a-6608a73c5824/r) shows the first invocation. You can see that the user’s initial query is passed directly to the retriever, which return suitable docs.

The invocation for followup question, [illustrated by this LangSmith trace](https://smith.langchain.com/public/7b463791-868b-42bd-8035-17b471e9c7cd/r) rephrases the user’s initial question to something more relevant to testing with LangSmith, resulting in higher quality docs.

And we now have a chatbot capable of conversational retrieval!

Next steps[​](#next-steps "Direct link to Next steps")
------------------------------------------------------

You now know how to build a conversational chatbot that can integrate past messages and domain-specific knowledge into its generations. There are many other optimizations you can make around this - check out the following pages for more information:

*   [Memory management](https://python.langchain.com/docs/use_cases/chatbots/memory_management/): This includes a guide on automatically updating chat history, as well as trimming, summarizing, or otherwise modifying long conversations to keep your bot focused.
*   [Retrieval](https://python.langchain.com/docs/use_cases/chatbots/retrieval/): A deeper dive into using different types of retrieval with your chatbot
*   [Tool usage](https://python.langchain.com/docs/use_cases/chatbots/tool_usage/): How to allows your chatbots to use tools that interact with other APIs and systems.