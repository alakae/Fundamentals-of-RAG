# WordParser - Microsoft Word Document Parser for RAG Systems

A Python utility for parsing Microsoft Word (.docx) files and preparing them for Retrieval-Augmented Generation (RAG) systems. Features intelligent chunking with paragraph, sentence, and word-boundary detection, plus rich metadata extraction from document properties.

## Overview

WordParser provides a robust interface to process Word documents for RAG applications. It extracts text content, document properties (title, author, timestamps), and intelligently splits content into overlapping chunks while respecting natural document structure.

## Features

- 📄 **DOCX Parsing**: Extract text from Microsoft Word documents (.docx format)
- ✂️ **Smart Chunking**: Hierarchical boundary detection (paragraphs → sentences → words)
- 📊 **Rich Metadata**: Extracts title, author, creation date, and modification date
- 🔄 **Configurable Overlap**: Maintain context continuity between chunks
- 🎯 **RAG-Ready**: Output format optimized for vector database ingestion
- 📝 **Paragraph Preservation**: Maintains document structure with double newlines
- ⚙️ **Customizable**: Adjust chunk size and overlap parameters

## Requirements

- Python 3.8+
- python-docx 1.2.0
- lxml 6.0.2

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install python-docx==1.2.0 lxml==6.0.2
```

## Project Structure

```
wordparser/
├── main.py            # DocxParser class and demo
├── requirements.txt   # Python dependencies
└── Sample.docx        # Sample Word document for testing
```

## Usage

### Basic Usage

```python
from main import DocxParser

# Initialize parser
parser = DocxParser(chunk_size=1000, chunk_overlap=200)

# Process a Word document
chunks = parser.process_document('document.docx')

# Access chunks
for chunk in chunks:
    print(f"Chunk {chunk['chunk_id']}: {chunk['text'][:100]}...")
    print(f"Author: {chunk['document_metadata']['author']}")
```

### Run the Demo

Ensure you have a file named `Sample.docx` in the directory, then run:

```bash
python main.py
```

Output:
```
--- Chunk 0 (Length: 487) ---
Introduction to Machine Learning

Machine learning is a subset of artificial intelligence...
Source: ML Tutorial
----------------------------------------------------------------

--- Chunk 1 (Length: 502) ---
...that focuses on the development of algorithms and statistical models...
Source: ML Tutorial
----------------------------------------------------------------
```

## DocxParser Class

### Constructor

```python
DocxParser(chunk_size=1000, chunk_overlap=200)
```

**Parameters:**
- `chunk_size` (int): Maximum characters per chunk (default: 1000)
- `chunk_overlap` (int): Characters to overlap between chunks (default: 200)

### Methods

#### `parse_docx(file_path: str) -> Dict`

Parses a Word document and extracts content with metadata.

```python
parser = DocxParser()
result = parser.parse_docx('document.docx')

# Returns:
# {
#     'content': 'Full document text with paragraphs...',
#     'paragraphs': ['Para 1', 'Para 2', ...],
#     'metadata': {
#         'filename': 'document.docx',
#         'file_path': '/absolute/path/document.docx',
#         'title': 'Document Title',
#         'author': 'John Doe',
#         'created': '2025-01-15 10:30:00',
#         'modified': '2025-11-18 14:22:00'
#     }
# }
```

#### `chunk_text(text: str) -> List[Dict]`

Splits text into overlapping chunks with intelligent boundary detection.

```python
parser = DocxParser(chunk_size=500, chunk_overlap=100)
chunks = parser.chunk_text("Your document text here...")

# Each chunk contains:
# {
#     'chunk_id': 0,
#     'text': 'Chunk content...',
#     'start_char': 0,
#     'end_char': 487,
#     'chunk_length': 487
# }
```

#### `process_document(file_path: str) -> List[Dict]`

Complete pipeline: parse DOCX and create chunks with full metadata.

```python
parser = DocxParser()
chunks = parser.process_document('document.docx')

# Each chunk includes both chunk data and document metadata
for chunk in chunks:
    print(chunk['text'])
    print(chunk['document_metadata'])
```

## Chunking Strategy

### Hierarchical Boundary Detection

WordParser uses a three-tier approach to find optimal chunk boundaries:

1. **Paragraph Breaks** (Priority 1): Looks for `\n\n` within 200 chars of target
2. **Sentence Endings** (Priority 2): Searches for `.`, `!`, `?` within 100 chars
3. **Word Boundaries** (Priority 3): Falls back to spaces within 50 chars
4. **Hard Split** (Fallback): Uses exact position if no better boundary found

This ensures:
- ✅ Respects document structure (paragraphs)
- ✅ Preserves sentence integrity
- ✅ Maintains context with overlap
- ✅ Handles edge cases gracefully

### Chunk Size Guidelines

| Chunk Size | Overlap | Best For |
|------------|---------|----------|
| 500 | 100 | Short documents, precise search |
| 1000 | 200 | **Balanced** (default); most documents |
| 1500 | 300 | Technical docs, longer context |
| 2000 | 400 | Reports, comprehensive passages |

### Paragraph Preservation

Paragraphs in the Word document are joined with double newlines (`\n\n`), preserving the document's logical structure for better chunking.

## Output Format

Each processed chunk includes:

```python
{
    'chunk_id': 0,                      # Sequential chunk number
    'text': 'Chunk content...',         # Actual text content
    'start_char': 0,                    # Starting character position
    'end_char': 487,                    # Ending character position
    'chunk_length': 487,                # Length in characters
    'document_metadata': {              # Word document properties
        'filename': 'document.docx',
        'file_path': '/path/to/document.docx',
        'title': 'My Document',
        'author': 'John Doe',
        'created': '2025-01-15 10:30:00',
        'modified': '2025-11-18 14:22:00'
    }
}
```

## Integration Examples

### ChromaDB Integration

```python
import chromadb
from main import DocxParser

# Process Word document
parser = DocxParser(chunk_size=1000, chunk_overlap=200)
chunks = parser.process_document('document.docx')

# Create ChromaDB collection
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("word_docs")

# Add chunks to ChromaDB
collection.add(
    ids=[f"{c['document_metadata']['filename']}_chunk_{c['chunk_id']}" 
         for c in chunks],
    documents=[c['text'] for c in chunks],
    metadatas=[{
        'filename': c['document_metadata']['filename'],
        'title': c['document_metadata']['title'],
        'author': c['document_metadata']['author'],
        'chunk_id': c['chunk_id']
    } for c in chunks]
)

# Query
results = collection.query(
    query_texts=["What is the main conclusion?"],
    n_results=5
)
```

### Save to JSON for RAG Ingestion

```python
import json
from main import DocxParser

parser = DocxParser()
chunks = parser.process_document('document.docx')

# Save for RAG ingestion
with open('word_chunks.json', 'w', encoding='utf-8') as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)
```

### Batch Processing Multiple Files

```python
from pathlib import Path
from main import DocxParser
import json

parser = DocxParser(chunk_size=1000, chunk_overlap=200)
all_chunks = []

# Process all .docx files in a directory
for docx_file in Path("documents/").glob("*.docx"):
    try:
        chunks = parser.process_document(str(docx_file))
        all_chunks.extend(chunks)
        
        title = chunks[0]['document_metadata']['title']
        print(f"✓ Processed {docx_file.name} ({title}): {len(chunks)} chunks")
    except Exception as e:
        print(f"✗ Error processing {docx_file.name}: {e}")

# Save combined output
with open("all_word_chunks.json", "w", encoding='utf-8') as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"\nTotal chunks: {len(all_chunks)}")
```

### LangChain Integration

```python
from main import DocxParser
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Process Word document
parser = DocxParser()
chunks = parser.process_document('document.docx')

# Convert to LangChain documents
documents = [
    Document(
        page_content=chunk['text'],
        metadata={
            'title': chunk['document_metadata']['title'],
            'author': chunk['document_metadata']['author'],
            'chunk_id': chunk['chunk_id'],
            'filename': chunk['document_metadata']['filename']
        }
    )
    for chunk in chunks
]

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Search
results = vectorstore.similarity_search("project timeline", k=5)
```

## Advanced Usage

### Custom Chunk Sizes

```python
# Small chunks for precise retrieval
small_parser = DocxParser(chunk_size=500, chunk_overlap=100)
small_chunks = small_parser.process_document('doc.docx')

# Large chunks for more context
large_parser = DocxParser(chunk_size=2000, chunk_overlap=400)
large_chunks = large_parser.process_document('doc.docx')
```

### Extract Metadata Only

```python
parser = DocxParser()
doc_data = parser.parse_docx('document.docx')

metadata = doc_data['metadata']
print(f"Title: {metadata['title']}")
print(f"Author: {metadata['author']}")
print(f"Created: {metadata['created']}")
print(f"Modified: {metadata['modified']}")
```

### Access Paragraphs Separately

```python
parser = DocxParser()
doc_data = parser.parse_docx('document.docx')

# Work with individual paragraphs
for i, para in enumerate(doc_data['paragraphs']):
    print(f"Paragraph {i}: {para[:100]}...")
```

### Custom Metadata Enrichment

```python
from datetime import datetime
from main import DocxParser

parser = DocxParser()
chunks = parser.process_document('document.docx')

# Add custom fields
for chunk in chunks:
    chunk['processed_at'] = datetime.now().isoformat()
    chunk['version'] = '1.0'
    chunk['language'] = 'en'
    
    # Add derived fields
    chunk['word_count'] = len(chunk['text'].split())
    chunk['paragraph_count'] = chunk['text'].count('\n\n') + 1
```

### Filter by Document Properties

```python
parser = DocxParser()

# Process multiple documents
all_chunks = []
for file in Path('docs/').glob('*.docx'):
    chunks = parser.process_document(str(file))
    all_chunks.extend(chunks)

# Filter by author
johns_docs = [c for c in all_chunks 
              if c['document_metadata']['author'] == 'John Doe']

# Filter by date
from datetime import datetime
recent_docs = [c for c in all_chunks 
               if c['document_metadata']['modified'] and
               datetime.fromisoformat(c['document_metadata']['modified']) > 
               datetime(2025, 1, 1)]
```

## Comparison with Other Parsers

| Feature | WordParser | PDFParser | TextParser |
|---------|-----------|-----------|------------|
| Input Format | .docx | .pdf | .txt |
| Dependencies | python-docx | PyPDF2 | None |
| Smart Boundaries | ✅ Para→Sent→Word | ✅ LangChain | ✅ Sent→Word |
| Metadata | Document props | Page info | File stats |
| Structure | Preserves paras | Page-based | Text-based |
| Use Case | Word docs | PDFs | Plain text |

## Troubleshooting

**Issue**: "FileNotFoundError: document.docx not found"
- Ensure file path is correct (relative or absolute)
- Check file exists and has `.docx` extension
- Verify read permissions

**Issue**: "BadZipFile: File is not a zip file"
- File may be corrupted
- Ensure it's actually a `.docx` file (not `.doc` legacy format)
- Try opening in Microsoft Word to verify integrity

**Issue**: "PackageNotFoundError: python-docx not installed"
- Install dependencies: `pip install -r requirements.txt`
- Or: `pip install python-docx`

**Issue**: Missing or incorrect metadata
- Document may not have properties set (title, author)
- Defaults to "Untitled" and "Unknown" if not present
- Edit document properties in Word to add metadata

**Issue**: Poor text extraction from tables
- python-docx extracts table text as paragraphs
- For complex tables, consider using `doc.tables` directly
- May need custom table parsing logic

**Issue**: Formatting lost
- WordParser extracts plain text only
- Bold, italic, colors are not preserved
- For formatted text, access `paragraph.runs` with styling

**Issue**: Images and shapes not extracted
- WordParser extracts text only
- Images ignored by default
- Use `doc.inline_shapes` for image processing

## Performance Considerations

- **Speed**: ~0.2-1 second per document (depends on size)
- **Memory**: Loads entire document into memory
- **Scalability**: Suitable for documents up to 50MB
- **Batch Processing**: Can efficiently process 100+ documents

For very large documents:
- Consider processing in sections
- Use streaming approaches for massive files
- Monitor memory usage during batch processing

## Limitations

- **Legacy .doc Format**: Only supports `.docx` (not `.doc`)
  - Use `doc2docx` or LibreOffice to convert old formats
- **Text Only**: No images, charts, or embedded objects
- **Complex Layouts**: Multi-column layouts may extract sequentially
- **Tables**: Basic table text extraction (no structure preservation)
- **Headers/Footers**: Not included in standard paragraph extraction
- **Comments/Revisions**: Track changes and comments not extracted

## Working with .doc Files

To convert legacy `.doc` files to `.docx`:

```bash
# Using LibreOffice command line
libreoffice --headless --convert-to docx document.doc

# Or use Python
pip install doc2docx
python -c "from doc2docx import convert; convert('document.doc')"
```

## Extracting Additional Elements

### Extract Tables

```python
from docx import Document

doc = Document('document.docx')

for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            print(cell.text, end='\t')
        print()
```

### Extract Headers and Footers

```python
from docx import Document

doc = Document('document.docx')

for section in doc.sections:
    print("Header:", section.header.paragraphs[0].text)
    print("Footer:", section.footer.paragraphs[0].text)
```

## Best Practices

1. **Verify File Format**: Ensure `.docx` not `.doc`
2. **Set Document Properties**: Add title/author in Word for better metadata
3. **Clean Documents**: Remove excessive formatting before parsing
4. **Test Chunk Sizes**: Adjust based on document structure
5. **Handle Errors**: Wrap in try-except for batch processing
6. **Preserve Context**: Use 15-20% overlap for chunk_size

## Next Steps

- Add support for tables and structured data
- Extract images with descriptions
- Include headers and footers
- Parse comments and track changes
- Add style/formatting preservation
- Implement section-aware chunking
- Create CLI interface with argparse
- Add support for .doc conversion

## Related Projects

See other demos in this repository:
- **pdfparser**: PDF document parser for RAG
- **textparser**: Plain text file parser
- **csvparser**: CSV to RAG document converter
- **ChromaDemo**: Vector database examples
- **BookSearch**: Complete RAG implementation

## License

Educational demo project.
