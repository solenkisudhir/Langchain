# Recursively split by character | 🦜️🔗 LangChain
This text splitter is the recommended one for generic text. It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is `["\n\n", "\n", " ", ""]`. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.

1.  How the text is split: by list of characters.
2.  How the chunk size is measured: by number of characters.

```
%pip install -qU langchain-text-splitters

```


```
# This is a long document we can split up.
with open("../../state_of_the_union.txt") as f:
    state_of_the_union = f.read()

```


```
from langchain_text_splitters import RecursiveCharacterTextSplitter

```


```
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

```


```
texts = text_splitter.create_documents([state_of_the_union])
print(texts[0])
print(texts[1])

```


```
page_content='Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and'
page_content='of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.'

```


```
text_splitter.split_text(state_of_the_union)[:2]

```


```
['Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and',
 'of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.']

```


Splitting text from languages without word boundaries[​](#splitting-text-from-languages-without-word-boundaries "Direct link to Splitting text from languages without word boundaries")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Some writing systems do not have [word boundaries](https://en.wikipedia.org/wiki/Category:Writing_systems_without_word_boundaries), for example Chinese, Japanese, and Thai. Splitting text with the default separator list of `["\n\n", "\n", " ", ""]` can cause words to be split between chunks. To keep words together, you can override the list of separators to include additional punctuation:

*   Add ASCII full-stop “`.`”, [Unicode fullwidth](https://en.wikipedia.org/wiki/Halfwidth_and_Fullwidth_Forms_(Unicode_block)) full stop “`．`” (used in Chinese text), and [ideographic full stop](https://en.wikipedia.org/wiki/CJK_Symbols_and_Punctuation) “`。`” (used in Japanese and Chinese)
*   Add [Zero-width space](https://en.wikipedia.org/wiki/Zero-width_space) used in Thai, Myanmar, Kmer, and Japanese.
*   Add ASCII comma “`,`”, Unicode fullwidth comma “`，`”, and Unicode ideographic comma “`、`”

```
text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
    # Existing args
)

```
