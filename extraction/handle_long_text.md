# Handle Long Text | 🦜️🔗 LangChain
When working with files, like PDFs, you’re likely to encounter text that exceeds your language model’s context window. To process this text, consider these strategies:

Keep in mind that these strategies have different trade off and the best strategy likely depends on the application that you’re designing!

We need some example data! Let’s download an article about [cars from wikipedia](https://en.wikipedia.org/wiki/Car) and load it as a LangChain `Document`.

Here, we’ll define schema to extract key developments from the text.

```
from typing import List, Optional

from langchain.chains import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class KeyDevelopment(BaseModel):
    """Information about a development in the history of cars."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.
    # Note that all fields are required rather than optional!
    year: int = Field(
        ..., description="The year when there was an important historic development."
    )
    description: str = Field(
        ..., description="What happened in this year? What was the development?"
    )
    evidence: str = Field(
        ...,
        description="Repeat in verbatim the sentence(s) from which the year and description information were extracted",
    )


class ExtractionData(BaseModel):
    """Extracted information about key developments in the history of cars."""

    key_developments: List[KeyDevelopment]


# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at identifying key historic development in text. "
            "Only extract important historic developments. Extract nothing if no important information can be found in the text.",
        ),
        # MessagesPlaceholder('examples'), # Keep on reading through this use case to see how to use examples to improve performance
        ("human", "{text}"),
    ]
)


# We will be using tool calling mode, which
# requires a tool calling capable model.
llm = ChatOpenAI(
    # Consider benchmarking with a good model to get
    # a sense of the best possible quality.
    model="gpt-4-0125-preview",
    # Remember to set the temperature to 0 for extractions!
    temperature=0,
)

extractor = prompt | llm.with_structured_output(
    schema=ExtractionData,
    method="function_calling",
    include_raw=False,
)

```


```
/home/eugene/.pyenv/versions/3.11.2/envs/langchain_3_11/lib/python3.11/site-packages/langchain_core/_api/beta_decorator.py:86: LangChainBetaWarning: The function `with_structured_output` is in beta. It is actively being worked on, so the API may change.
  warn_beta(

```


Split the documents into chunks such that each chunk fits into the context window of the LLMs.

After extracting data from across the chunks, we’ll want to merge the extractions together.

```
[KeyDevelopment(year=1966, description="The Toyota Corolla began production, recognized as the world's best-selling automobile.", evidence="The Toyota Corolla has been in production since 1966 and is recognized as the world's best-selling automobile."),
 KeyDevelopment(year=1769, description='Nicolas-Joseph Cugnot built the first steam-powered road vehicle.', evidence='French inventor Nicolas-Joseph Cugnot built the first steam-powered road vehicle in 1769.'),
 KeyDevelopment(year=1808, description='François Isaac de Rivaz designed and constructed the first internal combustion-powered automobile.', evidence='French-born Swiss inventor François Isaac de Rivaz designed and constructed the first internal combustion-powered automobile in 1808.'),
 KeyDevelopment(year=1886, description='Carl Benz patented his Benz Patent-Motorwagen, inventing the modern car.', evidence='The modern car—a practical, marketable automobile for everyday use—was invented in 1886, when German inventor Carl Benz patented his Benz Patent-Motorwagen.'),
 KeyDevelopment(year=1908, description='The 1908 Model T, an affordable car for the masses, was manufactured by the Ford Motor Company.', evidence='One of the first cars affordable by the masses was the 1908 Model T, an American car manufactured by the Ford Motor Company.'),
 KeyDevelopment(year=1881, description='Gustave Trouvé demonstrated a three-wheeled car powered by electricity.', evidence='In November 1881, French inventor Gustave Trouvé demonstrated a three-wheeled car powered by electricity at the International Exposition of Electricity.'),
 KeyDevelopment(year=1888, description="Bertha Benz undertook the first road trip by car to prove the road-worthiness of her husband's invention.", evidence="In August 1888, Bertha Benz, the wife of Carl Benz, undertook the first road trip by car, to prove the road-worthiness of her husband's invention."),
 KeyDevelopment(year=1896, description='Benz designed and patented the first internal-combustion flat engine, called boxermotor.', evidence='In 1896, Benz designed and patented the first internal-combustion flat engine, called boxermotor.'),
 KeyDevelopment(year=1897, description='Nesselsdorfer Wagenbau produced the Präsident automobil, one of the first factory-made cars in the world.', evidence='The first motor car in central Europe and one of the first factory-made cars in the world, was produced by Czech company Nesselsdorfer Wagenbau (later renamed to Tatra) in 1897, the Präsident automobil.'),
 KeyDevelopment(year=1890, description='Daimler Motoren Gesellschaft (DMG) was founded by Daimler and Maybach in Cannstatt.', evidence='Daimler and Maybach founded Daimler Motoren Gesellschaft (DMG) in Cannstatt in 1890.'),
 KeyDevelopment(year=1902, description='A new model DMG car was produced and named Mercedes after the Maybach engine.', evidence='Two years later, in 1902, a new model DMG car was produced and the model was named Mercedes after the Maybach engine, which generated 35 hp.'),
 KeyDevelopment(year=1891, description='Auguste Doriot and Louis Rigoulot completed the longest trip by a petrol-driven vehicle using a Daimler powered Peugeot Type 3.', evidence='In 1891, Auguste Doriot and his Peugeot colleague Louis Rigoulot completed the longest trip by a petrol-driven vehicle when their self-designed and built Daimler powered Peugeot Type 3 completed 2,100 kilometres (1,300 mi) from Valentigney to Paris and Brest and back again.'),
 KeyDevelopment(year=1895, description='George Selden was granted a US patent for a two-stroke car engine.', evidence='After a delay of 16 years and a series of attachments to his application, on 5 November 1895, Selden was granted a US patent (U.S. patent 549,160) for a two-stroke car engine.'),
 KeyDevelopment(year=1893, description='The first running, petrol-driven American car was built and road-tested by the Duryea brothers.', evidence='In 1893, the first running, petrol-driven American car was built and road-tested by the Duryea brothers of Springfield, Massachusetts.'),
 KeyDevelopment(year=1897, description='Rudolf Diesel built the first diesel engine.', evidence='In 1897, he built the first diesel engine.'),
 KeyDevelopment(year=1901, description='Ransom Olds started large-scale, production-line manufacturing of affordable cars at his Oldsmobile factory.', evidence='Large-scale, production-line manufacturing of affordable cars was started by Ransom Olds in 1901 at his Oldsmobile factory in Lansing, Michigan.'),
 KeyDevelopment(year=1913, description="Henry Ford began the world's first moving assembly line for cars at the Highland Park Ford Plant.", evidence="This concept was greatly expanded by Henry Ford, beginning in 1913 with the world's first moving assembly line for cars at the Highland Park Ford Plant."),
 KeyDevelopment(year=1914, description="Ford's assembly line worker could buy a Model T with four months' pay.", evidence="In 1914, an assembly line worker could buy a Model T with four months' pay."),
 KeyDevelopment(year=1926, description='Fast-drying Duco lacquer was developed, allowing for a variety of car colors.', evidence='Only Japan black would dry fast enough, forcing the company to drop the variety of colours available before 1913, until fast-drying Duco lacquer was developed in 1926.')]

```


Another simple idea is to chunk up the text, but instead of extracting information from every chunk, just focus on the the most relevant chunks.

Here’s a simple example that relies on the `FAISS` vectorstore.

In this case the RAG extractor is only looking at the top document.

```
year=1924 description="Germany's first mass-manufactured car, the Opel 4PS Laubfrosch, was produced, making Opel the top car builder in Germany with 37.5% of the market." evidence="Germany's first mass-manufactured car, the Opel 4PS Laubfrosch (Tree Frog), came off the line at Rüsselsheim in 1924, soon making Opel the top car builder in Germany, with 37.5 per cent of the market."
year=1925 description='Morris had 41% of total British car production, dominating the market.' evidence='in 1925, Morris had 41 per cent of total British car production.'
year=1925 description='Citroën, Renault, and Peugeot produced 550,000 cars in France, dominating the market.' evidence="Citroën did the same in France, coming to cars in 1919; between them and other cheap cars in reply such as Renault's 10CV and Peugeot's 5CV, they produced 550,000 cars in 1925."
year=2017 description='Production of petrol-fuelled cars peaked.' evidence='Production of petrol-fuelled cars peaked in 2017.'

```


Different methods have their own pros and cons related to cost, speed, and accuracy.