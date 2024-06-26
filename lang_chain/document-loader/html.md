# HTML | 🦜️🔗 LangChain
> [The HyperText Markup Language or HTML](https://en.wikipedia.org/wiki/HTML) is the standard markup language for documents designed to be displayed in a web browser.

This covers how to load `HTML` documents into a document format that we can use downstream.

```
from langchain_community.document_loaders import UnstructuredHTMLLoader

```


```
loader = UnstructuredHTMLLoader("example_data/fake-content.html")

```


```
    [Document(page_content='My First Heading\n\nMy first paragraph.', lookup_str='', metadata={'source': 'example_data/fake-content.html'}, lookup_index=0)]

```


Loading HTML with BeautifulSoup4[​](#loading-html-with-beautifulsoup4 "Direct link to Loading HTML with BeautifulSoup4")
------------------------------------------------------------------------------------------------------------------------

We can also use `BeautifulSoup4` to load HTML documents using the `BSHTMLLoader`. This will extract the text from the HTML into `page_content`, and the page title as `title` into `metadata`.

```
from langchain_community.document_loaders import BSHTMLLoader

```


```
loader = BSHTMLLoader("example_data/fake-content.html")
data = loader.load()
data

```


```
    [Document(page_content='\n\nTest Title\n\n\nMy First Heading\nMy first paragraph.\n\n\n', metadata={'source': 'example_data/fake-content.html', 'title': 'Test Title'})]

```


Loading HTML with SpiderLoader[​](#loading-html-with-spiderloader "Direct link to Loading HTML with SpiderLoader")
------------------------------------------------------------------------------------------------------------------

[Spider](https://spider.cloud/?ref=langchain) is the [fastest](https://github.com/spider-rs/spider/blob/main/benches/BENCHMARKS.md#benchmark-results) crawler. It converts any website into pure HTML, markdown, metadata or text while enabling you to crawl with custom actions using AI.

Spider allows you to use high performance proxies to prevent detection, caches AI actions, webhooks for crawling status, scheduled crawls etc...

Prerequisite[​](#prerequisite "Direct link to Prerequisite")
------------------------------------------------------------

You need to have a Spider api key to use this loader. You can get one on [spider.cloud](https://spider.cloud/).

```
%pip install --upgrade --quiet  langchain langchain-community spider-client

```


```
from langchain_community.document_loaders import SpiderLoader

loader = SpiderLoader(
    api_key="YOUR_API_KEY", url="https://spider.cloud", mode="crawl"
)

data = loader.load()

```


For guides and documentation, visit [Spider](https://spider.cloud/docs/api)

Loading HTML with FireCrawlLoader[​](#loading-html-with-firecrawlloader "Direct link to Loading HTML with FireCrawlLoader")
---------------------------------------------------------------------------------------------------------------------------

[FireCrawl](https://firecrawl.dev/?ref=langchain) crawls and convert any website into markdown. It crawls all accessible subpages and give you clean markdown and metadata for each.

FireCrawl handles complex tasks such as reverse proxies, caching, rate limits, and content blocked by JavaScript.

### Prerequisite[​](#prerequisite-1 "Direct link to Prerequisite")

You need to have a FireCrawl API key to use this loader. You can get one by signing up at [FireCrawl](https://firecrawl.dev/?ref=langchainpy).

```
%pip install --upgrade --quiet  langchain langchain-community firecrawl-py

from langchain_community.document_loaders import FireCrawlLoader


loader = FireCrawlLoader(
    api_key="YOUR_API_KEY", url="https://firecrawl.dev", mode="crawl"
)

data = loader.load()

```


For more information on how to use FireCrawl, visit [FireCrawl](https://firecrawl.dev/?ref=langchainpy).

Loading HTML with AzureAIDocumentIntelligenceLoader[​](#loading-html-with-azureaidocumentintelligenceloader "Direct link to Loading HTML with AzureAIDocumentIntelligenceLoader")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

[Azure AI Document Intelligence](https://aka.ms/doc-intelligence) (formerly known as `Azure Form Recognizer`) is machine-learning based service that extracts texts (including handwriting), tables, document structures (e.g., titles, section headings, etc.) and key-value-pairs from digital or scanned PDFs, images, Office and HTML files. Document Intelligence supports `PDF`, `JPEG/JPG`, `PNG`, `BMP`, `TIFF`, `HEIF`, `DOCX`, `XLSX`, `PPTX` and `HTML`.

This [current implementation](https://aka.ms/di-langchain) of a loader using `Document Intelligence` can incorporate content page-wise and turn it into LangChain documents. The default output format is markdown, which can be easily chained with `MarkdownHeaderTextSplitter` for semantic document chunking. You can also use `mode="single"` or `mode="page"` to return pure texts in a single page or document split by page.

### Prerequisite[​](#prerequisite-2 "Direct link to Prerequisite")

An Azure AI Document Intelligence resource in one of the 3 preview regions: **East US**, **West US2**, **West Europe** - follow [this document](https://learn.microsoft.com/azure/ai-services/document-intelligence/create-document-intelligence-resource?view=doc-intel-4.0.0) to create one if you don't have. You will be passing `<endpoint>` and `<key>` as parameters to the loader.

```
%pip install --upgrade --quiet  langchain langchain-community azure-ai-documentintelligence

from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader

file_path = "<filepath>"
endpoint = "<endpoint>"
key = "<key>"
loader = AzureAIDocumentIntelligenceLoader(
    api_endpoint=endpoint, api_key=key, file_path=file_path, api_model="prebuilt-layout"
)

documents = loader.load()

```
