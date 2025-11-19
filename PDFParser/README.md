# PDFParser - PDF to RAG Document Converter

A Python utility for extracting text from PDF files and converting them into chunked documents optimized for Retrieval-Augmented Generation (RAG) systems.

## Overview

PDFParser provides a simple interface to parse PDF documents, extract text content, and intelligently split it into overlapping chunks suitable for vector databases and semantic search applications. It uses PyPDF2 for text extraction and LangChain's text splitters for intelligent chunking.

## Features

- 📄 **PDF Text Extraction**: Extract text from single or multi-page PDFs
- ✂️ **Intelligent Chunking**: Split documents with configurable size and overlap
- 🔄 **Overlap Strategy**: Maintains context continuity between chunks
- 📊 **Structured Output**: Returns JSON-compatible dictionaries with metadata
- 💾 **Multiple Output Formats**: Save as JSON or formatted text files
- 🎯 **RAG-Ready**: Output optimized for vector database ingestion
- ⚙️ **Configurable**: Adjust chunk size and overlap for your use case

## Requirements

- Python 3.8+
- PyPDF2 3.0.1
- langchain-text-splitters 1.0.0

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install PyPDF2==3.0.1 langchain-text-splitters==1.0.0
```

## Project Structure

```
pdfparser/
├── pdf_parser.py       # PDFParser class implementation
├── main.py             # Usage examples and demo
├── requirements.txt    # Python dependencies
├── sample.pdf          # Sample PDF for testing
├── chunks.json         # Output JSON file (generated)
└── output.txt          # Output text file (generated)
```

## Usage

### Basic Usage

```python
from pdf_parser import PDFParser

# Initialize parser
parser = PDFParser(chunk_size=1000, chunk_overlap=200)

# Parse PDF and get structured results
results = parser.parse_pdf("document.pdf")

# Access chunks
for chunk in results:
    print(f"Chunk {chunk['chunk_id']}: {chunk['text'][:100]}...")
```

### Run the Demo

```bash
python main.py
```

Replace `"sample.pdf"` in `main.py` with your PDF file path.

### Save to JSON (RAG Ingestion)

```python
from pdf_parser import PDFParser
import json

parser = PDFParser(chunk_size=1000, chunk_overlap=200)
results = parser.parse_pdf("document.pdf")

# Save for RAG ingestion
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
```

### Save to Text File

```python
from pdf_parser import PDFParser

parser = PDFParser(chunk_size=1000, chunk_overlap=200)
parser.parse_pdf_to_file("document.pdf", "output.txt")
```

## PDFParser Class

### Constructor

```python
PDFParser(chunk_size=1000, chunk_overlap=200)
```

**Parameters:**
- `chunk_size` (int): Maximum characters per chunk (default: 1000)
- `chunk_overlap` (int): Characters to overlap between chunks (default: 200)

### Methods

#### `extract_text_from_pdf(pdf_path: str) -> str`

Extracts all text from a PDF file.

```python
parser = PDFParser()
text = parser.extract_text_from_pdf("document.pdf")
print(text)
```

#### `chunk_text(text: str) -> List[str]`

Splits text into overlapping chunks.

```python
parser = PDFParser(chunk_size=500, chunk_overlap=100)
chunks = parser.chunk_text("Your long text here...")
```

#### `parse_pdf(pdf_path: str) -> List[Dict[str, Any]]`

Parses PDF and returns structured chunks with metadata.

```python
parser = PDFParser()
results = parser.parse_pdf("document.pdf")

# Each result contains:
# {
#     "chunk_id": 0,
#     "text": "Chunk content...",
#     "source": "document.pdf",
#     "chunk_size": 950,
#     "total_chunks": 25
# }
```

#### `parse_pdf_to_file(pdf_path: str, output_path: str) -> None`

Parses PDF and saves formatted chunks to a text file.

```python
parser = PDFParser()
parser.parse_pdf_to_file("input.pdf", "output.txt")
```

## Output Format

### JSON Output (chunks.json)

```json
[
  {
    "chunk_id": 0,
    "text": "This is the content of the first chunk...",
    "source": "sample.pdf",
    "chunk_size": 987,
    "total_chunks": 15
  },
  {
    "chunk_id": 1,
    "text": "This is the content of the second chunk...",
    "source": "sample.pdf",
    "chunk_size": 1024,
    "total_chunks": 15
  }
]
```

### Text File Output (output.txt)

```
PDF Parser Results
Source: sample.pdf
Total Chunks: 15
================================================================================

Chunk 1/15
Size: 987 characters
--------------------------------------------------------------------------------
This is the content of the first chunk...

================================================================================

Chunk 2/15
Size: 1024 characters
--------------------------------------------------------------------------------
This is the content of the second chunk...
```

## Chunking Strategy

The parser uses LangChain's `RecursiveCharacterTextSplitter` with intelligent separators:

1. **Separator Hierarchy**: `"\n\n"` → `"\n"` → `" "` → `""`
2. **Context Preservation**: Overlap between chunks maintains continuity
3. **Boundary Respect**: Splits at natural breakpoints (paragraphs, sentences, words)

### Chunk Size Guidelines

| Chunk Size | Overlap | Best For |
|------------|---------|----------|
| 500 | 100 | Precise, granular search; short answers |
| 1000 | 200 | **Balanced** (default); general purpose |
| 1500 | 300 | More context; longer passages |
| 2000 | 400 | Maximum context; comprehensive answers |

## Integration Examples

### ChromaDB Integration

```python
import chromadb
from pdf_parser import PDFParser

# Parse PDF
parser = PDFParser(chunk_size=1000, chunk_overlap=200)
results = parser.parse_pdf("document.pdf")

# Create ChromaDB collection
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("pdf_docs")

# Add to ChromaDB
collection.add(
    ids=[f"chunk_{r['chunk_id']}" for r in results],
    documents=[r['text'] for r in results],
    metadatas=[{
        "source": r['source'],
        "chunk_id": r['chunk_id'],
        "chunk_size": r['chunk_size']
    } for r in results]
)

# Query
results = collection.query(
    query_texts=["What is the main topic?"],
    n_results=5
)
```

### LangChain Integration

```python
from pdf_parser import PDFParser
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Parse PDF
parser = PDFParser()
chunks = parser.parse_pdf("document.pdf")

# Convert to LangChain documents
documents = [
    Document(
        page_content=chunk['text'],
        metadata={
            "source": chunk['source'],
            "chunk_id": chunk['chunk_id']
        }
    )
    for chunk in chunks
]

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Search
results = vectorstore.similarity_search("query", k=5)
```

### Batch Processing Multiple PDFs

```python
from pdf_parser import PDFParser
from pathlib import Path
import json

parser = PDFParser(chunk_size=1000, chunk_overlap=200)
all_chunks = []

# Process all PDFs in a directory
for pdf_file in Path("pdfs/").glob("*.pdf"):
    try:
        chunks = parser.parse_pdf(str(pdf_file))
        all_chunks.extend(chunks)
        print(f"✓ Processed {pdf_file.name}: {len(chunks)} chunks")
    except Exception as e:
        print(f"✗ Error processing {pdf_file.name}: {e}")

# Save combined output
with open("all_chunks.json", "w") as f:
    json.dump(all_chunks, f, indent=2)

print(f"\nTotal chunks: {len(all_chunks)}")
```

## Advanced Usage

### Custom Chunking Parameters

```python
# Small chunks for precise retrieval
precise_parser = PDFParser(chunk_size=500, chunk_overlap=100)
small_chunks = precise_parser.parse_pdf("doc.pdf")

# Large chunks for more context
context_parser = PDFParser(chunk_size=2000, chunk_overlap=400)
large_chunks = context_parser.parse_pdf("doc.pdf")
```

### Extract Text Only

```python
from pdf_parser import PDFParser

parser = PDFParser()
full_text = parser.extract_text_from_pdf("document.pdf")

# Save full text without chunking
with open("full_text.txt", "w", encoding="utf-8") as f:
    f.write(full_text)
```

### Custom Metadata

```python
from pdf_parser import PDFParser

parser = PDFParser()
results = parser.parse_pdf("document.pdf")

# Enrich with custom metadata
for result in results:
    result["document_type"] = "technical_manual"
    result["date_processed"] = "2025-11-18"
    result["language"] = "en"
```

## Troubleshooting

**Issue**: "PDF file not found"
- Ensure the PDF file path is correct
- Use absolute paths if running from different directories
- Check file permissions

**Issue**: "Error reading PDF"
- PDF may be corrupted or password-protected
- Try opening the PDF manually to verify it's readable
- Some scanned PDFs may not have extractable text (requires OCR)

**Issue**: Poor text extraction quality
- Scanned PDFs need OCR (consider using pytesseract + pdf2image)
- Some PDFs have unusual encoding or formatting
- Try alternative libraries: pdfplumber, PyMuPDF (fitz)

**Issue**: Chunks are too large or too small
- Adjust `chunk_size` parameter based on your content
- Increase `chunk_overlap` for better context continuity
- Test different values to find optimal settings

**Issue**: Memory errors with large PDFs
- Process PDFs page-by-page instead of all at once
- Use streaming or batch processing
- Consider splitting large PDFs into smaller files

## Performance Considerations

- **Speed**: ~1-2 seconds per page for typical documents
- **Memory**: Loads entire PDF text into memory
- **Scaling**: For 100+ PDFs, consider parallel processing
- **Text Quality**: Depends on PDF source (digital vs. scanned)

## Limitations

- **Scanned PDFs**: Requires OCR preprocessing (not included)
- **Images/Tables**: Text extraction only; no image/table parsing
- **Complex Layouts**: Multi-column layouts may extract in unexpected order
- **Passwords**: Cannot process password-protected PDFs
- **Forms**: Form fields may not extract reliably

## Alternatives & Extensions

For more advanced PDF processing:
- **pdfplumber**: Better table extraction
- **PyMuPDF (fitz)**: Faster extraction, more features
- **pdf2image + pytesseract**: OCR for scanned documents
- **unstructured**: Unified API for multiple document types

## Next Steps

- Add OCR support for scanned PDFs
- Implement table extraction and parsing
- Add support for password-protected PDFs
- Include image extraction and description
- Create CLI interface with argparse
- Add page number tracking in metadata
- Implement parallel processing for multiple files

## Related Projects

See other demos in this repository:
- **csvparser**: CSV to RAG document converter
- **textparser**: Plain text file parser
- **ChromaDemo**: Vector database with ChromaDB
- **BookSearch**: Full RAG implementation

## License

Educational demo project.
