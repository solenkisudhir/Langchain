# Handle Files | 🦜️🔗 LangChain
Besides raw text data, you may wish to extract information from other file types such as PowerPoint presentations or PDFs.

You can use LangChain [document loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/) to parse files into a text format that can be fed into LLMs.

Here, we’ll be looking at MIME-type based parsing which is often useful for extraction based applications if you’re writing server code that accepts user uploaded files.

In this case, it’s best to assume that the file extension of the file provided by the user is wrong and instead infer the mimetype from the binary content of the file.

Let’s download some content. This will be an HTML file, but the code below will work with other file types.

```
import magic
from langchain.document_loaders.parsers import BS4HTMLParser, PDFMinerParser
from langchain.document_loaders.parsers.generic import MimeTypeBasedParser
from langchain.document_loaders.parsers.txt import TextParser
from langchain_community.document_loaders import Blob

# Configure the parsers that you want to use per mime-type!
HANDLERS = {
    "application/pdf": PDFMinerParser(),
    "text/plain": TextParser(),
    "text/html": BS4HTMLParser(),
}

# Instantiate a mimetype based parser with the given parsers
MIMETYPE_BASED_PARSER = MimeTypeBasedParser(
    handlers=HANDLERS,
    fallback_parser=None,
)

mime = magic.Magic(mime=True)
mime_type = mime.from_buffer(data)

# A blob represents binary data by either reference (path on file system)
# or value (bytes in memory).
blob = Blob.from_data(
    data=data,
    mime_type=mime_type,
)

parser = HANDLERS[mime_type]
documents = parser.parse(blob=blob)

```
