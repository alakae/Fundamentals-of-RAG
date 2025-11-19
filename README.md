# RAG Demos - Retrieval-Augmented Generation Examples

A comprehensive collection of Python utilities and demonstrations for building Retrieval-Augmented Generation (RAG) systems. This repository contains parsers, chunkers, vector databases, and complete RAG implementations to help you understand and build your own RAG applications.

## 🎯 Overview

This repository provides practical examples and reusable tools for every stage of the RAG pipeline:
- **Document Parsing**: Extract text from PDFs, Word docs, CSV, and plain text
- **Text Chunking**: Multiple strategies for splitting documents intelligently
- **Vector Storage**: Examples with ChromaDB and FAISS
- **Semantic Search**: BM25, vector search, and hybrid approaches
- **Complete RAG**: End-to-end implementations with Ollama and LLMs

## 📁 Repository Structure

### 🔍 Complete RAG Systems

#### **BookSearch/**
Full-featured RAG implementations with progressive complexity.

**Contains:**
- `app_v1.py` - Basic RAG demo with in-memory documents
- `app_v2.py` - File-based RAG with ingestion pipeline  
- `hybrid_rag.py` - Hybrid search (BM25 + semantic vector search with RRF)

**Key Features:**
- Ollama integration (llama3.3 + nomic-embed-text)
- ChromaDB vector storage
- Automatic document chunking with overlap
- Source citation in answers
- Reciprocal Rank Fusion (RRF) for hybrid search

**Use Case:** Question-answering over book collections (Shakespeare, Sherlock Holmes, Frankenstein, etc.)

[📖 Full Documentation](./BookSearch/README.md)

---

### 💾 Vector Database Demos

#### **ChromaDemo/**
Minimal ChromaDB implementation for semantic search.

**Contains:**
- `ingest_and_query.py` - Ingest text files and perform semantic queries

**Key Features:**
- Persistent ChromaDB storage
- Automatic chunking (1500 chars, 200 overlap)
- Sentence-transformers embeddings (all-MiniLM-L6-v2)
- Metadata filtering
- Idempotent ingestion

**Use Case:** Simple semantic search over text document collections

[📖 Full Documentation](./ChromaDemo/README.md)

---

### 📄 Document Parsers

#### **PDFParser/**
Extract and chunk text from PDF documents for RAG ingestion.

**Contains:**
- `pdf_parser.py` - PDFParser class implementation
- `main.py` - Usage examples

**Key Features:**
- PyPDF2-based text extraction
- LangChain text splitter integration
- Intelligent boundary detection
- Structured JSON output with metadata
- Multiple output formats (JSON, text)

**Dependencies:** PyPDF2, langchain-text-splitters

[📖 Full Documentation](./PDFParser/README.md)

---

#### **WordParser/**
Parse Microsoft Word (.docx) documents for RAG systems.

**Contains:**
- `main.py` - DocxParser class and demo

**Key Features:**
- Extracts text from .docx files
- Document properties (title, author, dates)
- Hierarchical chunking (paragraph → sentence → word)
- Preserves paragraph structure
- MD5-based document IDs

**Dependencies:** python-docx, lxml

[📖 Full Documentation](./WordParser/README.md)

---

#### **TextParser/**
Parse plain text files with smart chunking.

**Contains:**
- `main.py` - TextDocumentParser class and demo

**Key Features:**
- Zero dependencies (Python stdlib only)
- Sentence-aware boundary detection
- Rich file metadata extraction
- Configurable chunk size and overlap
- MD5 document IDs

**Use Case:** Processing plain text documents, logs, markdown files

[📖 Full Documentation](./TextParser/README.md)

---

#### **CSVParser/**
Convert CSV data to RAG-ready JSON documents.

**Contains:**
- `main.py` - CSV to JSON converter
- Sample datasets (sample_data.csv, big_data.csv)

**Key Features:**
- Transforms rows into searchable text
- Preserves all columns as metadata
- JSON output for vector database ingestion
- No external dependencies

**Use Case:** Making structured data semantically searchable

[📖 Full Documentation](./CSVParser/README.md)

---

### ✂️ Text Chunking

#### **ChunkingDemo/**
Comprehensive document chunker with 8 different strategies.

**Contains:**
- `document_chunker.py` - CLI tool with multiple chunking methods

**Chunking Methods:**
1. **Line-by-Line**: Group by number of lines
2. **Fixed Size**: Fixed character chunks with overlap
3. **Sliding Window**: Overlapping windows
4. **Sentence-Based**: Group by sentences
5. **Paragraph-Based**: Group by paragraphs
6. **Page-Based**: Simulate pages (by line count)
7. **Section-Based**: Split on headings (markdown, etc.)
8. **Token-Based**: BERT tokenizer-based chunking

**Supports:** TXT, PDF, DOC, DOCX files

**Dependencies:** PyPDF2, python-docx, transformers (optional)

[📖 Full Documentation](./ChunkingDemo/README.md)

---

### 🔎 Search Examples

#### **SearchTool/**
Jupyter notebooks demonstrating different search approaches.

**Contains:**
- `BM25-vs-Semantic.ipynb` - Comparison of keyword vs vector search
- `SearchDemo.ipynb` - Search implementation examples
- `Semantic-Demo.ipynb` - Semantic search demonstration

**Key Topics:**
- BM25 keyword search
- Vector embeddings and similarity
- Hybrid search strategies
- Search quality comparison

**Use Case:** Understanding search approaches for RAG

---

#### **SearchFiles/**
Collection of classic literature texts for search and RAG testing.

**Contains:** 18 classic books in plain text format
- Adventures of Huckleberry Finn
- Adventures of Sherlock Holmes
- Alice in Wonderland
- Beowulf
- Complete Works of William Shakespeare
- Dracula
- Frankenstein
- Great Gatsby
- Jane Eyre
- Moby Dick
- Pride and Prejudice
- And more...

**Use Case:** Test corpus for RAG and search implementations

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Basic Installation

Most demos have their own `requirements.txt`. Install per project:

```bash
cd BookSearch
pip install -r requirements.txt
```

### Quick Examples

**1. Basic RAG with BookSearch:**
```bash
cd BookSearch
python app_v2.py init          # Check setup
python app_v2.py ingest        # Ingest documents
python app_v2.py ask "Who is Sherlock Holmes?"
```

**2. ChromaDB Semantic Search:**
```bash
cd ChromaDemo
pip install -r requirements.txt
python ingest_and_query.py
```

**3. Parse a PDF:**
```bash
cd PDFParser
pip install -r requirements.txt
python main.py
```

**4. Parse a Word Document:**
```bash
cd WordParser
pip install -r requirements.txt
python main.py
```

**5. Chunk a Document:**
```bash
cd ChunkingDemo
pip install -r requirements.txt
python document_chunker.py sample_document.txt sentence --max-sentences 5
```

**6. Convert CSV to RAG Format:**
```bash
cd CSVParser
python main.py
```

## 🏗️ Building Your Own RAG System

### Step-by-Step Approach

1. **Choose Your Parser** (based on document format)
   - PDF → PDFParser
   - Word → WordParser
   - Plain text → TextParser
   - Structured data → CSVParser

2. **Select Chunking Strategy**
   - Small chunks (300-500 chars) → Precise retrieval
   - Medium chunks (1000-1500 chars) → Balanced
   - Large chunks (2000+ chars) → Maximum context
   - Use ChunkingDemo to experiment

3. **Pick Vector Database**
   - ChromaDB → Easy setup, great for prototypes
   - FAISS → High performance, production-ready
   - Pinecone/Weaviate → Managed, scalable

4. **Implement Search**
   - Semantic only → Simple, fast
   - Hybrid (BM25 + Vector) → Best quality
   - See SearchTool notebooks for comparisons

5. **Add LLM Generation**
   - Ollama → Local, free
   - OpenAI → High quality
   - Anthropic → Long context

### Architecture Pattern

```python
# 1. Parse documents
from pdfparser.main import PDFParser
parser = PDFParser(chunk_size=1000, chunk_overlap=200)
chunks = parser.process_document('document.pdf')

# 2. Store in vector DB
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("docs")
collection.add(
    ids=[f"chunk_{c['chunk_id']}" for c in chunks],
    documents=[c['text'] for c in chunks],
    metadatas=[c['document_metadata'] for c in chunks]
)

# 3. Query
results = collection.query(
    query_texts=["What is the main topic?"],
    n_results=5
)

# 4. Generate answer with LLM
import ollama
context = "\n\n".join(results['documents'][0])
prompt = f"Answer based on context:\n{context}\n\nQuestion: What is the main topic?"
answer = ollama.generate(model="llama3.3", prompt=prompt)
print(answer['response'])
```

## 🔧 Common Configurations

### Chunk Size Guidelines

| Document Type | Chunk Size | Overlap | Rationale |
|--------------|------------|---------|-----------|
| Technical docs | 800-1200 | 150-250 | Balance detail & context |
| Books/Articles | 1000-1500 | 200-300 | Preserve narrative flow |
| Code documentation | 500-800 | 100-150 | Precise code examples |
| Chat logs | 300-500 | 50-100 | Short exchanges |
| Research papers | 1500-2000 | 300-400 | Complex arguments |

### Embedding Models

| Model | Dimensions | Speed | Best For |
|-------|-----------|-------|----------|
| all-MiniLM-L6-v2 | 384 | Very Fast | General purpose |
| nomic-embed-text | 768 | Fast | Ollama integration |
| text-embedding-ada-002 | 1536 | Medium | High quality (OpenAI) |
| instructor-large | 768 | Slow | Domain-specific |

## 📊 Comparison Matrix

| Feature | BookSearch | ChromaDemo | PDFParser | WordParser | TextParser | CSVParser |
|---------|-----------|------------|-----------|------------|------------|-----------|
| Complete RAG | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Vector DB | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| LLM Integration | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Hybrid Search | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| PDF Support | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Word Support | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Structured Data | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Zero Dependencies | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.8+**: Primary language
- **ChromaDB**: Vector database
- **Ollama**: Local LLM inference
- **PyPDF2**: PDF parsing
- **python-docx**: Word document parsing
- **LangChain**: Text splitting utilities

### Optional Dependencies
- **rank-bm25**: BM25 search algorithm
- **transformers**: BERT tokenizer for chunking
- **sentence-transformers**: Embedding models

## 📚 Learning Path

### Beginner
1. Start with **ChromaDemo** to understand vector databases
2. Explore **TextParser** for basic document processing
3. Try **ChunkingDemo** to experiment with chunking strategies

### Intermediate
4. Use **PDFParser** and **WordParser** for real documents
5. Study **BookSearch/app_v1.py** and **app_v2.py** for RAG basics
6. Review **SearchTool** notebooks for search concepts

### Advanced
7. Implement **hybrid_rag.py** for production-quality search
8. Build custom parsers combining multiple demos
9. Scale to production with proper error handling and monitoring

## 🤝 Use Cases

### Documentation Q&A
- **Tools**: PDFParser + BookSearch
- **Example**: Company policy documents, technical manuals

### Customer Support
- **Tools**: ChromaDemo + hybrid search
- **Example**: FAQ database, support ticket history

### Research Assistant
- **Tools**: PDFParser + BookSearch
- **Example**: Academic papers, research notes

### Code Documentation
- **Tools**: TextParser + semantic search
- **Example**: README files, code comments

### Data Analysis
- **Tools**: CSVParser + vector search
- **Example**: Customer databases, log analysis

## 🐛 Troubleshooting

### Common Issues

**"Module not found" errors**
```bash
pip install -r requirements.txt
```

**Ollama connection errors (BookSearch)**
```bash
ollama serve  # Start Ollama server
ollama pull llama3.3
ollama pull nomic-embed-text
```

**PDF extraction issues**
- Scanned PDFs need OCR (pytesseract + pdf2image)
- Password-protected PDFs not supported
- Try alternative: pdfplumber or PyMuPDF

**Memory errors with large files**
- Reduce chunk size
- Process files in batches
- Use streaming approaches

**Poor search results**
- Adjust chunk size (try smaller/larger)
- Increase overlap (15-20% of chunk size)
- Use hybrid search instead of semantic only

## 🔒 Best Practices

1. **Start Simple**: Begin with ChromaDemo, then add complexity
2. **Test Chunking**: Use ChunkingDemo to find optimal strategy
3. **Version Control**: Track changes to chunk size and overlap
4. **Monitor Quality**: Regularly evaluate retrieval accuracy
5. **Document Metadata**: Always preserve source information
6. **Error Handling**: Wrap parsers in try-except for production
7. **Batch Processing**: Process multiple files efficiently
8. **Clean Data**: Preprocess documents before ingestion

## 📖 Additional Resources

### Related Projects
- [LangChain](https://github.com/langchain-ai/langchain) - RAG framework
- [LlamaIndex](https://github.com/run-llama/llama_index) - Data framework for LLMs
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector database
- [Ollama](https://github.com/ollama/ollama) - Local LLM runtime

### Learning Materials
- [RAG Explanation](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Vector Database Guide](https://www.pinecone.io/learn/vector-database/)
- [Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)

## 🤝 Contributing

This is an educational repository. Feel free to:
- Fork and modify for your projects
- Report issues or suggest improvements
- Share your implementations and learnings

## 📝 License

Educational demo project. All code provided as-is for learning purposes.

## 🙏 Acknowledgments

Sample texts in SearchFiles/ are public domain works from Project Gutenberg.

---

**Happy Building! 🚀**

Start with any demo that matches your needs, or combine multiple parsers for a complete solution. Each subdirectory has detailed documentation to guide you.
