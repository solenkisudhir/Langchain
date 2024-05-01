# Sequences: Chaining runnables | 🦜️🔗 LangChain
One key advantage of the `Runnable` interface is that any two runnables can be “chained” together into sequences. The output of the previous runnable’s `.invoke()` call is passed as input to the next runnable. This can be done using the pipe operator (`|`), or the more explicit `.pipe()` method, which does the same thing. The resulting `RunnableSequence` is itself a runnable, which means it can be invoked, streamed, or piped just like any other runnable.

The pipe operator[​](#the-pipe-operator "Direct link to The pipe operator")
---------------------------------------------------------------------------

To show off how this works, let’s go through an example. We’ll walk through a common pattern in LangChain: using a [prompt template](https://python.langchain.com/docs/modules/model_io/prompts/) to format input into a [chat model](https://python.langchain.com/docs/modules/model_io/chat/), and finally converting the chat message output into a string with an [output parser](https://python.langchain.com/docs/modules/model_io/output_parsers/).

```
%pip install --upgrade --quiet langchain langchain-anthropic

```


```
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
model = ChatAnthropic(model_name="claude-3-haiku-20240307")

chain = prompt | model | StrOutputParser()

```


Prompts and models are both runnable, and the output type from the prompt call is the same as the input type of the chat model, so we can chain them together. We can then invoke the resulting sequence like any other runnable:

```
chain.invoke({"topic": "bears"})

```


```
"Here's a bear joke for you:\n\nWhy don't bears wear socks? \nBecause they have bear feet!\n\nHow's that? I tried to keep it light and silly. Bears can make for some fun puns and jokes. Let me know if you'd like to hear another one!"

```


### Coercion[​](#coercion "Direct link to Coercion")

We can even combine this chain with more runnables to create another chain. This may involve some input/output formatting using other types of runnables, depending on the required inputs and outputs of the chain components.

For example, let’s say we wanted to compose the joke generating chain with another chain that evaluates whether or not the generated joke was funny.

We would need to be careful with how we format the input into the next chain. In the below example, the dict in the chain is automatically parsed and converted into a [`RunnableParallel`](https://python.langchain.com/docs/expression_language/primitives/parallel/), which runs all of its values in parallel and returns a dict with the results.

This happens to be the same format the next prompt template expects. Here it is in action:

```
from langchain_core.output_parsers import StrOutputParser

analysis_prompt = ChatPromptTemplate.from_template("is this a funny joke? {joke}")

composed_chain = {"joke": chain} | analysis_prompt | model | StrOutputParser()

```


```
composed_chain.invoke({"topic": "bears"})

```


```
"That's a pretty classic and well-known bear pun joke. Whether it's considered funny is quite subjective, as humor is very personal. Some people may find that type of pun-based joke amusing, while others may not find it that humorous. Ultimately, the funniness of a joke is in the eye (or ear) of the beholder. If you enjoyed the joke and got a chuckle out of it, then that's what matters most."

```


Functions will also be coerced into runnables, so you can add custom logic to your chains too. The below chain results in the same logical flow as before:

```
composed_chain_with_lambda = (
    chain
    | (lambda input: {"joke": input})
    | analysis_prompt
    | model
    | StrOutputParser()
)

```


```
composed_chain_with_lambda.invoke({"topic": "beets"})

```


```
'I appreciate the effort, but I have to be honest - I didn\'t find that joke particularly funny. Beet-themed puns can be quite hit-or-miss, and this one falls more on the "miss" side for me. The premise is a bit too straightforward and predictable. While I can see the logic behind it, the punchline just doesn\'t pack much of a comedic punch. \n\nThat said, I do admire your willingness to explore puns and wordplay around vegetables. Cultivating a good sense of humor takes practice, and not every joke is going to land. The important thing is to keep experimenting and finding what works. Maybe try for a more unexpected or creative twist on beet-related humor next time. But thanks for sharing - I always appreciate when humans test out jokes on me, even if they don\'t always make me laugh out loud.'

```


However, keep in mind that using functions like this may interfere with operations like streaming. See [this section](https://python.langchain.com/docs/expression_language/primitives/functions/) for more information.

The `.pipe()` method[​](#the-.pipe-method "Direct link to the-.pipe-method")
----------------------------------------------------------------------------

We could also compose the same sequence using the `.pipe()` method. Here’s what that looks like:

```
from langchain_core.runnables import RunnableParallel

composed_chain_with_pipe = (
    RunnableParallel({"joke": chain})
    .pipe(analysis_prompt)
    .pipe(model)
    .pipe(StrOutputParser())
)

```


```
composed_chain_with_pipe.invoke({"topic": "battlestar galactica"})

```


```
'That\'s a pretty good Battlestar Galactica-themed pun! I appreciated the clever play on words with "Centurion" and "center on." It\'s the kind of nerdy, science fiction-inspired humor that fans of the show would likely enjoy. The joke is clever and demonstrates a good understanding of the Battlestar Galactica universe. I\'d be curious to hear any other Battlestar-related jokes you might have up your sleeve. As long as they don\'t reproduce copyrighted material, I\'m happy to provide my thoughts on the humor and appeal for fans of the show.'

```
