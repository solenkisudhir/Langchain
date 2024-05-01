# Composition | 🦜️🔗 LangChain
LangChain provides a user friendly interface for composing different parts of prompts together. You can do this with either string prompts or chat prompts. Constructing prompts this way allows for easy reuse of components.

String prompt composition[​](#string-prompt-composition "Direct link to String prompt composition")
---------------------------------------------------------------------------------------------------

When working with string prompts, each template is joined together. You can work with either prompts directly or strings (the first element in the list needs to be a prompt).

```
from langchain_core.prompts import PromptTemplate

```


```
prompt = (
    PromptTemplate.from_template("Tell me a joke about {topic}")
    + ", make it funny"
    + "\n\nand in {language}"
)

```


```
PromptTemplate(input_variables=['language', 'topic'], template='Tell me a joke about {topic}, make it funny\n\nand in {language}')

```


```
prompt.format(topic="sports", language="spanish")

```


```
'Tell me a joke about sports, make it funny\n\nand in spanish'

```


You can also use it in an LLMChain, just like before.

```
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

```


```
chain = LLMChain(llm=model, prompt=prompt)

```


```
chain.run(topic="sports", language="spanish")

```


```
'¿Por qué el futbolista llevaba un paraguas al partido?\n\nPorque pronosticaban lluvia de goles.'

```


Chat prompt composition[​](#chat-prompt-composition "Direct link to Chat prompt composition")
---------------------------------------------------------------------------------------------

A chat prompt is made up a of a list of messages. Purely for developer experience, we’ve added a convenient way to create these prompts. In this pipeline, each new element is a new message in the final prompt.

```
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

```


First, let’s initialize the base ChatPromptTemplate with a system message. It doesn’t have to start with a system, but it’s often good practice

```
prompt = SystemMessage(content="You are a nice pirate")

```


You can then easily create a pipeline combining it with other messages _or_ message templates. Use a `Message` when there is no variables to be formatted, use a `MessageTemplate` when there are variables to be formatted. You can also use just a string (note: this will automatically get inferred as a HumanMessagePromptTemplate.)

```
new_prompt = (
    prompt + HumanMessage(content="hi") + AIMessage(content="what?") + "{input}"
)

```


Under the hood, this creates an instance of the ChatPromptTemplate class, so you can use it just as you did before!

```
new_prompt.format_messages(input="i said hi")

```


```
[SystemMessage(content='You are a nice pirate', additional_kwargs={}),
 HumanMessage(content='hi', additional_kwargs={}, example=False),
 AIMessage(content='what?', additional_kwargs={}, example=False),
 HumanMessage(content='i said hi', additional_kwargs={}, example=False)]

```


You can also use it in an LLMChain, just like before.

```
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

```


```
chain = LLMChain(llm=model, prompt=new_prompt)

```


```
'Oh, hello! How can I assist you today?'

```


Using PipelinePrompt[​](#using-pipelineprompt "Direct link to Using PipelinePrompt")
------------------------------------------------------------------------------------

LangChain includes an abstraction [PipelinePromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.pipeline.PipelinePromptTemplate.html), which can be useful when you want to reuse parts of prompts. A PipelinePrompt consists of two main parts:

*   Final prompt: The final prompt that is returned
*   Pipeline prompts: A list of tuples, consisting of a string name and a prompt template. Each prompt template will be formatted and then passed to future prompt templates as a variable with the same name.

```
from langchain_core.prompts.pipeline import PipelinePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

```


```
full_template = """{introduction}

{example}

{start}"""
full_prompt = PromptTemplate.from_template(full_template)

```


```
introduction_template = """You are impersonating {person}."""
introduction_prompt = PromptTemplate.from_template(introduction_template)

```


```
example_template = """Here's an example of an interaction:

Q: {example_q}
A: {example_a}"""
example_prompt = PromptTemplate.from_template(example_template)

```


```
start_template = """Now, do this for real!

Q: {input}
A:"""
start_prompt = PromptTemplate.from_template(start_template)

```


```
input_prompts = [
    ("introduction", introduction_prompt),
    ("example", example_prompt),
    ("start", start_prompt),
]
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_prompt, pipeline_prompts=input_prompts
)

```


```
pipeline_prompt.input_variables

```


```
['example_q', 'person', 'input', 'example_a']

```


```
print(
    pipeline_prompt.format(
        person="Elon Musk",
        example_q="What's your favorite car?",
        example_a="Tesla",
        input="What's your favorite social media site?",
    )
)

```


```
You are impersonating Elon Musk.

Here's an example of an interaction:

Q: What's your favorite car?
A: Tesla

Now, do this for real!

Q: What's your favorite social media site?
A:

```
