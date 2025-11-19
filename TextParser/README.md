# TextParser - Text Document Parser for RAG Systems

A Python utility for parsing plain text files and preparing them for Retrieval-Augmented Generation (RAG) systems. Features intelligent chunking with sentence-aware boundary detection and comprehensive metadata extraction.

## Overview

TextParser provides a clean interface to process text documents for RAG applications. It reads text files, extracts metadata, and intelligently splits content into overlapping chunks while respecting natural language boundaries like sentences and paragraphs.

## Features

- 📄 **Text File Processing**: Parse plain text files with UTF-8 encoding
- ✂️ **Smart Chunking**: Sentence and word-aware boundary detection
- 🔄 **Configurable Overlap**: Maintain context between adjacent chunks
- 📊 **Rich Metadata**: Automatic extraction of file properties and statistics
- 🆔 **Document IDs**: MD5-based unique identifiers for each document
- 🎯 **RAG-Ready**: Output format optimized for vector database ingestion
- ⚙️ **Customizable**: Adjust chunk size and overlap parameters

## Requirements

- Python 3.8+
- No external dependencies (uses only Python standard library)

## Installation

No installation needed! TextParser uses only Python's standard library:

```bash
python --version  # Ensure Python 3.8+
```

## Project Structure

```
textparser/
├── main.py           # TextDocumentParser class and demo
└── sample_doc.txt    # Sample text file for testing
```

## Usage

### Basic Usage

```python
from main import TextDocumentParser

# Initialize parser
parser = TextDocumentParser(chunk_size=1000, chunk_overlap=200)

# Process a text file
chunks = parser.process_document('document.txt')

# Access chunks
for chunk in chunks:
    print(f"Chunk {chunk['chunk_id']}: {chunk['text'][:100]}...")
    print(f"Metadata: {chunk['document_metadata']}")
```

### Run the Demo

```bash
python main.py
```

Output:
```
Document: sample_doc.txt
Total chunks: 3

Chunk 0:
Length: 487 chars
Text: The Squirrel and the Wi-Fi Router

Once upon a time, in a quiet suburban neighborhood...

Chunk 1:
Length: 502 chars
Text: ...trying to connect to the strongest Wi-Fi signal in town...

Chunk 2:
Length: 318 chars
Text: ...The moral of the story: curiosity may crash your connection, but it also connects you to laughter.
```

## TextDocumentParser Class

### Constructor

```python
TextDocumentParser(chunk_size=1000, chunk_overlap=200)
```

**Parameters:**
- `chunk_size` (int): Maximum characters per chunk (default: 1000)
- `chunk_overlap` (int): Characters to overlap between chunks (default: 200)

### Methods

#### `parse_file(file_path: str) -> Dict`

Parses a text file and extracts content with metadata.

```python
parser = TextDocumentParser()
result = parser.parse_file('document.txt')

# Returns:
# {
#     'content': 'Full document text...',
#     'metadata': {
#         'filename': 'document.txt',
#         'file_path': '/absolute/path/document.txt',
#         'file_size': 5432,
#         'file_extension': '.txt',
#         'document_id': 'a1b2c3d4e5f6...',
#         'char_count': 5234,
#         'word_count': 876
#     }
# }
```

#### `chunk_text(text: str) -> List[Dict]`

Splits text into overlapping chunks with smart boundary detection.

```python
parser = TextDocumentParser(chunk_size=500, chunk_overlap=100)
chunks = parser.chunk_text("Your long text here...")

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

Complete pipeline: parse file and create chunks with metadata.

```python
parser = TextDocumentParser()
chunks = parser.process_document('document.txt')

# Each chunk contains both chunk data and document metadata
for chunk in chunks:
    print(chunk['text'])
    print(chunk['document_metadata'])
```

## Chunking Strategy

### Smart Boundary Detection

TextParser uses a hierarchical approach to find optimal chunk boundaries:

1. **Sentence Endings**: Looks for `.`, `!`, `?`, or `\n\n` within 100 chars of target
2. **Word Boundaries**: Falls back to spaces within 50 chars of target
3. **Hard Split**: Uses exact position only if no better boundary found

This ensures chunks:
- ✅ End at natural break points
- ✅ Preserve sentence integrity
- ✅ Maintain context with overlap

### Chunk Size Guidelines

| Chunk Size | Overlap | Best For |
|------------|---------|----------|
| 300-500 | 50-100 | Short texts, precise retrieval |
| 500-1000 | 100-200 | **Balanced** (default) |
| 1000-1500 | 200-300 | Longer context, narrative text |
| 1500-2000 | 300-400 | Maximum context preservation |

### Example Chunking Behavior

```python
text = "First sentence. Second sentence. Third sentence. Fourth sentence."
parser = TextDocumentParser(chunk_size=40, chunk_overlap=10)
chunks = parser.chunk_text(text)

# Chunk 0: "First sentence. Second sentence."
# Chunk 1: "Second sentence. Third sentence."
# Chunk 2: "Third sentence. Fourth sentence."
```

## Output Format

Each processed chunk includes:

```python
{
    'chunk_id': 0,                      # Sequential chunk number
    'text': 'Chunk content...',         # Actual text content
    'start_char': 0,                    # Starting character position
    'end_char': 487,                    # Ending character position
    'chunk_length': 487,                # Length in characters
    'document_metadata': {              # Source document info
        'filename': 'document.txt',
        'file_path': '/path/to/document.txt',
        'file_size': 5432,
        'file_extension': '.txt',
        'document_id': 'a1b2c3d4e5f6...',
        'char_count': 5234,
        'word_count': 876
    }
}
```

## Integration Examples

### ChromaDB Integration

```python
import chromadb
from main import TextDocumentParser

# Process text file
parser = TextDocumentParser(chunk_size=1000, chunk_overlap=200)
chunks = parser.process_document('document.txt')

# Create ChromaDB collection
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("text_docs")

# Add chunks to ChromaDB
collection.add(
    ids=[f"doc_{c['document_metadata']['document_id']}_chunk_{c['chunk_id']}" 
         for c in chunks],
    documents=[c['text'] for c in chunks],
    metadatas=[{
        'filename': c['document_metadata']['filename'],
        'chunk_id': c['chunk_id'],
        'start_char': c['start_char'],
        'end_char': c['end_char']
    } for c in chunks]
)

# Query
results = collection.query(
    query_texts=["What is the main topic?"],
    n_results=5
)
```

### Save to JSON for RAG Ingestion

```python
import json
from main import TextDocumentParser

parser = TextDocumentParser()
chunks = parser.process_document('document.txt')

# Save for RAG ingestion
with open('chunks.json', 'w', encoding='utf-8') as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)
```

### Batch Processing Multiple Files

```python
from pathlib import Path
from main import TextDocumentParser

parser = TextDocumentParser(chunk_size=1000, chunk_overlap=200)
all_chunks = []

# Process all .txt files in a directory
for txt_file in Path("documents/").glob("*.txt"):
    try:
        chunks = parser.process_document(str(txt_file))
        all_chunks.extend(chunks)
        print(f"✓ Processed {txt_file.name}: {len(chunks)} chunks")
    except Exception as e:
        print(f"✗ Error processing {txt_file.name}: {e}")

print(f"\nTotal chunks: {len(all_chunks)}")
```

### LangChain Integration

```python
from main import TextDocumentParser
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Process text file
parser = TextDocumentParser()
chunks = parser.process_document('document.txt')

# Convert to LangChain documents
documents = [
    Document(
        page_content=chunk['text'],
        metadata={
            'filename': chunk['document_metadata']['filename'],
            'chunk_id': chunk['chunk_id'],
            'start_char': chunk['start_char']
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

## Advanced Usage

### Custom Chunk Sizes

```python
# Small chunks for precise retrieval
small_parser = TextDocumentParser(chunk_size=300, chunk_overlap=50)
small_chunks = small_parser.process_document('doc.txt')

# Large chunks for more context
large_parser = TextDocumentParser(chunk_size=2000, chunk_overlap=400)
large_chunks = large_parser.process_document('doc.txt')
```

### Parse Without Chunking

```python
parser = TextDocumentParser()
doc_data = parser.parse_file('document.txt')

# Access full content
full_text = doc_data['content']
metadata = doc_data['metadata']

print(f"Document: {metadata['filename']}")
print(f"Words: {metadata['word_count']}")
print(f"Characters: {metadata['char_count']}")
```

### Custom Metadata Enrichment

```python
from datetime import datetime
from main import TextDocumentParser

parser = TextDocumentParser()
chunks = parser.process_document('document.txt')

# Add custom fields to each chunk
for chunk in chunks:
    chunk['processed_at'] = datetime.now().isoformat()
    chunk['version'] = '1.0'
    chunk['language'] = 'en'
    
    # Add derived fields
    chunk['word_count'] = len(chunk['text'].split())
    chunk['sentence_count'] = chunk['text'].count('.') + \
                              chunk['text'].count('!') + \
                              chunk['text'].count('?')
```

### Filter and Process Chunks

```python
parser = TextDocumentParser()
chunks = parser.process_document('document.txt')

# Filter out very short chunks
filtered_chunks = [c for c in chunks if c['chunk_length'] > 100]

# Process only chunks containing specific keywords
relevant_chunks = [c for c in chunks if 'RAG' in c['text'] or 'retrieval' in c['text'].lower()]

# Split into introduction, body, and conclusion
total_chars = chunks[-1]['end_char']
intro = [c for c in chunks if c['end_char'] < total_chars * 0.2]
body = [c for c in chunks if total_chars * 0.2 <= c['start_char'] < total_chars * 0.8]
conclusion = [c for c in chunks if c['start_char'] >= total_chars * 0.8]
```

## Comparison with Other Parsers

| Feature | TextParser | PDFParser | CSVParser |
|---------|-----------|-----------|-----------|
| Input Format | .txt files | .pdf files | .csv files |
| Dependencies | None | PyPDF2, LangChain | None |
| Smart Boundaries | ✅ Sentence-aware | ✅ LangChain splitter | ❌ Row-based |
| Metadata | File stats | Page/chunk info | Column data |
| Use Case | Plain text | Documents | Structured data |

## Troubleshooting

**Issue**: "File not found"
- Check file path is correct (relative or absolute)
- Ensure file has read permissions
- Verify file exists: `os.path.exists('file.txt')`

**Issue**: Encoding errors
- TextParser uses UTF-8 by default
- For other encodings, modify: `open(file, 'r', encoding='latin-1')`
- Common encodings: `utf-8`, `latin-1`, `ascii`, `cp1252`

**Issue**: Chunks are too large or too small
- Adjust `chunk_size` parameter
- Increase overlap for better context: `chunk_overlap=300`
- Check sentence structure of your documents

**Issue**: Poor boundary detection
- Documents with few sentence endings may use word boundaries
- Very short sentences may cause many small chunks
- Consider adjusting boundary search range in `chunk_text()`

**Issue**: Missing metadata
- Ensure file is accessible before calling `parse_file()`
- Check file permissions: `os.access('file.txt', os.R_OK)`

## Performance Considerations

- **Speed**: ~0.1-0.5 seconds for typical documents (1000-5000 words)
- **Memory**: Loads entire file into memory
- **Scalability**: Suitable for files up to several MB
- **Batch Processing**: Can process 100+ files efficiently

For very large files (>50MB):
- Consider streaming approaches
- Process in chunks without loading full file
- Use generators for memory efficiency

## Best Practices

1. **Choose Appropriate Chunk Size**: Match to your retrieval granularity needs
2. **Use Sufficient Overlap**: 15-20% of chunk_size maintains good context
3. **Validate Encoding**: Test with sample files before batch processing
4. **Preserve Metadata**: Document ID helps track chunks back to source
5. **Test Boundary Detection**: Review chunks to ensure quality splits

## Extending TextParser

### Add Custom Parsing Logic

```python
class CustomTextParser(TextDocumentParser):
    def parse_file(self, file_path: str) -> Dict:
        result = super().parse_file(file_path)
        
        # Add custom metadata
        content = result['content']
        result['metadata']['line_count'] = content.count('\n')
        result['metadata']['avg_word_length'] = (
            sum(len(word) for word in content.split()) / 
            len(content.split())
        )
        
        return result
```

### Custom Chunking Strategy

```python
class ParagraphParser(TextDocumentParser):
    def chunk_text(self, text: str) -> List[Dict]:
        # Split by paragraphs instead
        paragraphs = text.split('\n\n')
        chunks = []
        
        for i, para in enumerate(paragraphs):
            if para.strip():
                chunks.append({
                    'chunk_id': i,
                    'text': para.strip(),
                    'chunk_length': len(para)
                })
        
        return chunks
```

## Next Steps

- Add support for multiple file formats (markdown, HTML, RTF)
- Implement parallel processing for multiple files
- Add language detection and multilingual support
- Create streaming API for very large files
- Add semantic-aware chunking using NLP
- Implement deduplication for repeated content
- Add CLI interface with argparse

## Related Projects

See other demos in this repository:
- **pdfparser**: PDF document parser for RAG
- **csvparser**: CSV to RAG document converter
- **wordparser**: Word document parser
- **ChromaDemo**: Vector database examples
- **BookSearch**: Complete RAG implementation

## License

Educational demo project.
