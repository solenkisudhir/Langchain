# Chains | 🦜️🔗 LangChain
Chains refer to sequences of calls - whether to an LLM, a tool, or a data preprocessing step. The primary supported way to do this is with [LCEL](https://python.langchain.com/docs/expression_language/).

LCEL is great for constructing your own chains, but it’s also nice to have chains that you can use off-the-shelf. There are two types of off-the-shelf chains that LangChain supports:

*   Chains that are built with LCEL. In this case, LangChain offers a higher-level constructor method. However, all that is being done under the hood is constructing a chain with LCEL.
    
*   \[Legacy\] Chains constructed by subclassing from a legacy `Chain` class. These chains do not use LCEL under the hood but are rather standalone classes.
    

We are working creating methods that create LCEL versions of all chains. We are doing this for a few reasons.

1.  Chains constructed in this way are nice because if you want to modify the internals of a chain you can simply modify the LCEL.
    
2.  These chains natively support streaming, async, and batch out of the box.
    
3.  These chains automatically get observability at each step.
    

This page contains two lists. First, a list of all LCEL chain constructors. Second, a list of all legacy Chains.

LCEL Chains[​](#lcel-chains "Direct link to LCEL Chains")
---------------------------------------------------------

Below is a table of all LCEL chain constructors. In addition, we report on:

**Chain Constructor**

The constructor function for this chain. These are all methods that return LCEL runnables. We also link to the API documentation.

**Function Calling**

Whether this requires OpenAI function calling.

**Other Tools**

What other tools (if any) are used in this chain.

**When to Use**

Our commentary on when to use this chain.



* Chain Constructor: create_stuff_documents_chain
  * Function Calling: 
  * Other Tools: 
  * When to Use: This chain takes a list of documents and formats them all into a prompt, then passes that prompt to an LLM. It passes ALL documents, so you should make sure it fits within the context window the LLM you are using.
* Chain Constructor: create_openai_fn_runnable
  * Function Calling: ✅
  * Other Tools: 
  * When to Use: If you want to use OpenAI function calling to OPTIONALLY structured an output response. You may pass in multiple functions for it call, but it does not have to call it.
* Chain Constructor: create_structured_output_runnable
  * Function Calling: ✅
  * Other Tools: 
  * When to Use: If you want to use OpenAI function calling to FORCE the LLM to respond with a certain function. You may only pass in one function, and the chain will ALWAYS return this response.
* Chain Constructor: load_query_constructor_runnable
  * Function Calling: 
  * Other Tools: 
  * When to Use: Can be used to generate queries. You must specify a list of allowed operations, and then will return a runnable that converts a natural language query into those allowed operations.
* Chain Constructor: create_sql_query_chain
  * Function Calling: 
  * Other Tools: SQL Database
  * When to Use: If you want to construct a query for a SQL database from natural language.
* Chain Constructor: create_history_aware_retriever
  * Function Calling: 
  * Other Tools: Retriever
  * When to Use: This chain takes in conversation history and then uses that to generate a search query which is passed to the underlying retriever.
* Chain Constructor: create_retrieval_chain
  * Function Calling: 
  * Other Tools: Retriever
  * When to Use: This chain takes in a user inquiry, which is then passed to the retriever to fetch relevant documents. Those documents (and original inputs) are then passed to an LLM to generate a response


Legacy Chains[​](#legacy-chains "Direct link to Legacy Chains")
---------------------------------------------------------------

Below we report on the legacy chain types that exist. We will maintain support for these until we are able to create a LCEL alternative. We report on:

**Chain**

Name of the chain, or name of the constructor method. If constructor method, this will return a `Chain` subclass.

**Function Calling**

Whether this requires OpenAI Function Calling.

**Other Tools**

Other tools used in the chain.

**When to Use**

Our commentary on when to use.



* Chain: APIChain
  * Function Calling: 
  * Other Tools: Requests Wrapper
  * When to Use: This chain uses an LLM to convert a query into an API request, then executes that request, gets back a response, and then passes that request to an LLM to respond
* Chain: OpenAPIEndpointChain
  * Function Calling: 
  * Other Tools: OpenAPI Spec
  * When to Use: Similar to APIChain, this chain is designed to interact with APIs. The main difference is this is optimized for ease of use with OpenAPI endpoints
* Chain: ConversationalRetrievalChain
  * Function Calling: 
  * Other Tools: Retriever
  * When to Use: This chain can be used to have conversations with a document. It takes in a question and (optional) previous conversation history. If there is previous conversation history, it uses an LLM to rewrite the conversation into a query to send to a retriever (otherwise it just uses the newest user input). It then fetches those documents and passes them (along with the conversation) to an LLM to respond.
* Chain: StuffDocumentsChain
  * Function Calling: 
  * Other Tools: 
  * When to Use: This chain takes a list of documents and formats them all into a prompt, then passes that prompt to an LLM. It passes ALL documents, so you should make sure it fits within the context window the LLM you are using.
* Chain: ReduceDocumentsChain
  * Function Calling: 
  * Other Tools: 
  * When to Use: This chain combines documents by iterative reducing them. It groups documents into chunks (less than some context length) then passes them into an LLM. It then takes the responses and continues to do this until it can fit everything into one final LLM call. Useful when you have a lot of documents, you want to have the LLM run over all of them, and you can do in parallel.
* Chain: MapReduceDocumentsChain
  * Function Calling: 
  * Other Tools: 
  * When to Use: This chain first passes each document through an LLM, then reduces them using the ReduceDocumentsChain. Useful in the same situations as ReduceDocumentsChain, but does an initial LLM call before trying to reduce the documents.
* Chain: RefineDocumentsChain
  * Function Calling: 
  * Other Tools: 
  * When to Use: This chain collapses documents by generating an initial answer based on the first document and then looping over the remaining documents to refine its answer. This operates sequentially, so it cannot be parallelized. It is useful in similar situatations as MapReduceDocuments Chain, but for cases where you want to build up an answer by refining the previous answer (rather than parallelizing calls).
* Chain: MapRerankDocumentsChain
  * Function Calling: 
  * Other Tools: 
  * When to Use: This calls on LLM on each document, asking it to not only answer but also produce a score of how confident it is. The answer with the highest confidence is then returned. This is useful when you have a lot of documents, but only want to answer based on a single document, rather than trying to combine answers (like Refine and Reduce methods do).
* Chain: ConstitutionalChain
  * Function Calling: 
  * Other Tools: 
  * When to Use: This chain answers, then attempts to refine its answer based on constitutional principles that are provided. Use this when you want to enforce that a chain’s answer follows some principles.
* Chain: LLMChain
  * Function Calling: 
  * Other Tools: 
  * When to Use: 
* Chain: ElasticsearchDatabaseChain
  * Function Calling: 
  * Other Tools: ElasticSearch Instance
  * When to Use: This chain converts a natural language question to an ElasticSearch query, and then runs it, and then summarizes the response. This is useful for when you want to ask natural language questions of an Elastic Search database
* Chain: FlareChain
  * Function Calling: 
  * Other Tools: 
  * When to Use: This implements FLARE, an advanced retrieval technique. It is primarily meant as an exploratory advanced retrieval method.
* Chain: ArangoGraphQAChain
  * Function Calling: 
  * Other Tools: Arango Graph
  * When to Use: This chain constructs an Arango query from natural language, executes that query against the graph, and then passes the results back to an LLM to respond.
* Chain: GraphCypherQAChain
  * Function Calling: 
  * Other Tools: A graph that works with Cypher query language
  * When to Use: This chain constructs an Cypher query from natural language, executes that query against the graph, and then passes the results back to an LLM to respond.
* Chain: FalkorDBGraphQAChain
  * Function Calling: 
  * Other Tools: Falkor Database
  * When to Use: This chain constructs a FalkorDB query from natural language, executes that query against the graph, and then passes the results back to an LLM to respond.
* Chain: HugeGraphQAChain
  * Function Calling: 
  * Other Tools: HugeGraph
  * When to Use: This chain constructs an HugeGraph query from natural language, executes that query against the graph, and then passes the results back to an LLM to respond.
* Chain: KuzuQAChain
  * Function Calling: 
  * Other Tools: Kuzu Graph
  * When to Use: This chain constructs a Kuzu Graph query from natural language, executes that query against the graph, and then passes the results back to an LLM to respond.
* Chain: NebulaGraphQAChain
  * Function Calling: 
  * Other Tools: Nebula Graph
  * When to Use: This chain constructs a Nebula Graph query from natural language, executes that query against the graph, and then passes the results back to an LLM to respond.
* Chain: NeptuneOpenCypherQAChain
  * Function Calling: 
  * Other Tools: Neptune Graph
  * When to Use: This chain constructs an Neptune Graph query from natural language, executes that query against the graph, and then passes the results back to an LLM to respond.
* Chain: GraphSparqlChain
  * Function Calling: 
  * Other Tools: Graph that works with SparQL
  * When to Use: This chain constructs an SparQL query from natural language, executes that query against the graph, and then passes the results back to an LLM to respond.
* Chain: LLMMath
  * Function Calling: 
  * Other Tools: 
  * When to Use: This chain converts a user question to a math problem and then executes it (using numexpr)
* Chain: LLMCheckerChain
  * Function Calling: 
  * Other Tools: 
  * When to Use: This chain uses a second LLM call to varify its initial answer. Use this when you to have an extra layer of validation on the initial LLM call.
* Chain: LLMSummarizationChecker
  * Function Calling: 
  * Other Tools: 
  * When to Use: This chain creates a summary using a sequence of LLM calls to make sure it is extra correct. Use this over the normal summarization chain when you are okay with multiple LLM calls (eg you care more about accuracy than speed/cost).
* Chain: create_citation_fuzzy_match_chain
  * Function Calling: ✅
  * Other Tools: 
  * When to Use: Uses OpenAI function calling to answer questions and cite its sources.
* Chain: create_extraction_chain
  * Function Calling: ✅
  * Other Tools: 
  * When to Use: Uses OpenAI Function calling to extract information from text.
* Chain: create_extraction_chain_pydantic
  * Function Calling: ✅
  * Other Tools: 
  * When to Use: Uses OpenAI function calling to extract information from text into a Pydantic model. Compared to create_extraction_chain this has a tighter integration with Pydantic.
* Chain: get_openapi_chain
  * Function Calling: ✅
  * Other Tools: OpenAPI Spec
  * When to Use: Uses OpenAI function calling to query an OpenAPI.
* Chain: create_qa_with_structure_chain
  * Function Calling: ✅
  * Other Tools: 
  * When to Use: Uses OpenAI function calling to do question answering over text and respond in a specific format.
* Chain: create_qa_with_sources_chain
  * Function Calling: ✅
  * Other Tools: 
  * When to Use: Uses OpenAI function calling to answer questions with citations.
* Chain: QAGenerationChain
  * Function Calling: 
  * Other Tools: 
  * When to Use: Creates both questions and answers from documents. Can be used to generate question/answer pairs for evaluation of retrieval projects.
* Chain: RetrievalQAWithSourcesChain
  * Function Calling: 
  * Other Tools: Retriever
  * When to Use: Does question answering over retrieved documents, and cites it sources. Use this when you want the answer response to have sources in the text response. Use this over load_qa_with_sources_chain when you want to use a retriever to fetch the relevant document as part of the chain (rather than pass them in).
* Chain: load_qa_with_sources_chain
  * Function Calling: 
  * Other Tools: Retriever
  * When to Use: Does question answering over documents you pass in, and cites it sources. Use this when you want the answer response to have sources in the text response. Use this over RetrievalQAWithSources when you want to pass in the documents directly (rather than rely on a retriever to get them).
* Chain: RetrievalQA
  * Function Calling: 
  * Other Tools: Retriever
  * When to Use: This chain first does a retrieval step to fetch relevant documents, then passes those documents into an LLM to generate a response.
* Chain: MultiPromptChain
  * Function Calling: 
  * Other Tools: 
  * When to Use: This chain routes input between multiple prompts. Use this when you have multiple potential prompts you could use to respond and want to route to just one.
* Chain: MultiRetrievalQAChain
  * Function Calling: 
  * Other Tools: Retriever
  * When to Use: This chain routes input between multiple retrievers. Use this when you have multiple potential retrievers you could fetch relevant documents from and want to route to just one.
* Chain: EmbeddingRouterChain
  * Function Calling: 
  * Other Tools: 
  * When to Use: This chain uses embedding similarity to route incoming queries.
* Chain: LLMRouterChain
  * Function Calling: 
  * Other Tools: 
  * When to Use: This chain uses an LLM to route between potential options.
* Chain: load_summarize_chain
  * Function Calling: 
  * Other Tools: 
  * When to Use: 
* Chain: LLMRequestsChain
  * Function Calling: 
  * Other Tools: 
  * When to Use: This chain constructs a URL from user input, gets data at that URL, and then summarizes the response. Compared to APIChain, this chain is not focused on a single API spec but is more general
