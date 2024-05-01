# Output Parsers | 🦜️🔗 LangChain
Output parsers are responsible for taking the output of an LLM and transforming it to a more suitable format. This is very useful when you are using LLMs to generate any form of structured data.

Besides having a large collection of different types of output parsers, one distinguishing benefit of LangChain OutputParsers is that many of them support streaming.

[Quick Start](https://python.langchain.com/docs/modules/model_io/output_parsers/quick_start/)[​](#quick-start "Direct link to quick-start")
-------------------------------------------------------------------------------------------------------------------------------------------

See [this quick-start guide](https://python.langchain.com/docs/modules/model_io/output_parsers/quick_start/) for an introduction to output parsers and how to work with them.

Output Parser Types[​](#output-parser-types "Direct link to Output Parser Types")
---------------------------------------------------------------------------------

LangChain has lots of different types of output parsers. This is a list of output parsers LangChain supports. The table below has various pieces of information:

**Name**: The name of the output parser

**Supports Streaming**: Whether the output parser supports streaming.

**Has Format Instructions**: Whether the output parser has format instructions. This is generally available except when (a) the desired schema is not specified in the prompt but rather in other parameters (like OpenAI function calling), or (b) when the OutputParser wraps another OutputParser.

**Calls LLM**: Whether this output parser itself calls an LLM. This is usually only done by output parsers that attempt to correct misformatted output.

**Input Type**: Expected input type. Most output parsers work on both strings and messages, but some (like OpenAI Functions) need a message with specific kwargs.

**Output Type**: The output type of the object returned by the parser.

**Description**: Our commentary on this output parser and when to use it.



* Name: OpenAITools
  * Supports Streaming: 
  * Has Format Instructions: (Passes tools to model)
  * Calls LLM: 
  * Input Type: Message (with tool_choice)
  * Output Type: JSON object
  * Description: Uses latest OpenAI function calling args tools and tool_choice to structure the return output. If you are using a model that supports function calling, this is generally the most reliable method.
* Name: OpenAIFunctions
  * Supports Streaming: ✅
  * Has Format Instructions: (Passes functions to model)
  * Calls LLM: 
  * Input Type: Message (with function_call)
  * Output Type: JSON object
  * Description: Uses legacy OpenAI function calling args functions and function_call to structure the return output.
* Name: JSON
  * Supports Streaming: ✅
  * Has Format Instructions: ✅
  * Calls LLM: 
  * Input Type: str | Message
  * Output Type: JSON object
  * Description: Returns a JSON object as specified. You can specify a Pydantic model and it will return JSON for that model. Probably the most reliable output parser for getting structured data that does NOT use function calling.
* Name: XML
  * Supports Streaming: ✅
  * Has Format Instructions: ✅
  * Calls LLM: 
  * Input Type: str | Message
  * Output Type: dict
  * Description: Returns a dictionary of tags. Use when XML output is needed. Use with models that are good at writing XML (like Anthropic's).
* Name: CSV
  * Supports Streaming: ✅
  * Has Format Instructions: ✅
  * Calls LLM: 
  * Input Type: str | Message
  * Output Type: List[str]
  * Description: Returns a list of comma separated values.
* Name: OutputFixing
  * Supports Streaming: 
  * Has Format Instructions: 
  * Calls LLM: ✅
  * Input Type: str | Message
  * Output Type: 
  * Description: Wraps another output parser. If that output parser errors, then this will pass the error message and the bad output to an LLM and ask it to fix the output.
* Name: RetryWithError
  * Supports Streaming: 
  * Has Format Instructions: 
  * Calls LLM: ✅
  * Input Type: str | Message
  * Output Type: 
  * Description: Wraps another output parser. If that output parser errors, then this will pass the original inputs, the bad output, and the error message to an LLM and ask it to fix it. Compared to OutputFixingParser, this one also sends the original instructions.
* Name: Pydantic
  * Supports Streaming: 
  * Has Format Instructions: ✅
  * Calls LLM: 
  * Input Type: str | Message
  * Output Type: pydantic.BaseModel
  * Description: Takes a user defined Pydantic model and returns data in that format.
* Name: YAML
  * Supports Streaming: 
  * Has Format Instructions: ✅
  * Calls LLM: 
  * Input Type: str | Message
  * Output Type: pydantic.BaseModel
  * Description: Takes a user defined Pydantic model and returns data in that format. Uses YAML to encode it.
* Name: PandasDataFrame
  * Supports Streaming: 
  * Has Format Instructions: ✅
  * Calls LLM: 
  * Input Type: str | Message
  * Output Type: dict
  * Description: Useful for doing operations with pandas DataFrames.
* Name: Enum
  * Supports Streaming: 
  * Has Format Instructions: ✅
  * Calls LLM: 
  * Input Type: str | Message
  * Output Type: Enum
  * Description: Parses response into one of the provided enum values.
* Name: Datetime
  * Supports Streaming: 
  * Has Format Instructions: ✅
  * Calls LLM: 
  * Input Type: str | Message
  * Output Type: datetime.datetime
  * Description: Parses response into a datetime string.
* Name: Structured
  * Supports Streaming: 
  * Has Format Instructions: ✅
  * Calls LLM: 
  * Input Type: str | Message
  * Output Type: Dict[str, str]
  * Description: An output parser that returns structured information. It is less powerful than other output parsers since it only allows for fields to be strings. This can be useful when you are working with smaller LLMs.
