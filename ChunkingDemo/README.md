# Document Chunker for RAG Systems

A comprehensive Python tool for chunking documents using various strategies, designed specifically for Retrieval-Augmented Generation (RAG) systems.

## Features

- **Multiple Chunking Strategies**: 8 different chunking methods
- **File Format Support**: TXT, PDF, DOC, DOCX
- **Command-line Interface**: Easy to use from terminal
- **Flexible Parameters**: Customizable chunk sizes and overlap
- **Metadata Preservation**: Maintains chunk context and positioning
- **Token-based Chunking**: Uses BERT tokenizer for accurate token counting

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make the script executable:
```bash
chmod +x document_chunker.py
```

## Supported Chunking Methods

### 1. Line-by-Line Chunking
Chunks text by grouping a specified number of lines together.

```bash
python document_chunker.py sample_document.txt line --max-lines 5
```

### 2. Fixed-Size Chunking
Chunks text into fixed character-size segments with optional overlap.

```bash
python document_chunker.py sample_document.txt fixed --chunk-size 500 --overlap 50
```

### 3. Sliding Window Chunking
Uses a sliding window approach with configurable window size and step size.

```bash
python document_chunker.py sample_document.txt sliding --window-size 800 --step-size 400
```

### 4. Sentence-Based Chunking
Chunks text by grouping sentences together.

```bash
python document_chunker.py sample_document.txt sentence --max-sentences 3
```

### 5. Paragraph-Based Chunking
Chunks text by grouping paragraphs together.

```bash
python document_chunker.py sample_document.txt paragraph --max-paragraphs 2
```

### 6. Page-Based Chunking
Simulates page-based chunking by grouping lines (useful for PDFs).

```bash
python document_chunker.py sample_document.txt page --lines-per-page 30
```

### 7. Section/Heading-Based Chunking
Chunks text by sections defined by headings (supports Markdown-style headers).

```bash
python document_chunker.py sample_document.txt section --heading-pattern "^#{1,6}\\s+"
```

### 8. Token-Based Chunking
Chunks text by token count using BERT tokenizer.

```bash
python document_chunker.py sample_document.txt token --max-tokens 256
```

## Command-Line Usage

```bash
python document_chunker.py <file> <method> [options]
```

### Arguments

- `file`: Path to the document file
- `method`: Chunking method (`line`, `fixed`, `sliding`, `sentence`, `paragraph`, `page`, `section`, `token`)

### Options

#### General Options
- `--no-metadata`: Hide metadata in output
- `--output FILE`: Save chunks to file instead of printing to terminal

#### Method-Specific Options

**Line Chunking:**
- `--max-lines N`: Maximum lines per chunk (default: 10)

**Fixed-Size Chunking:**
- `--chunk-size N`: Chunk size in characters (default: 1000)
- `--overlap N`: Overlap between chunks (default: 0)

**Sliding Window:**
- `--window-size N`: Window size (default: 1000)
- `--step-size N`: Step size (default: 500)

**Sentence Chunking:**
- `--max-sentences N`: Maximum sentences per chunk (default: 5)

**Paragraph Chunking:**
- `--max-paragraphs N`: Maximum paragraphs per chunk (default: 3)

**Page Chunking:**
- `--lines-per-page N`: Lines per page (default: 50)

**Section Chunking:**
- `--heading-pattern REGEX`: Regex pattern for headings (default: `^#{1,6}\\s+`)

**Token Chunking:**
- `--max-tokens N`: Maximum tokens per chunk (default: 512)

## Examples

### Basic Usage
```bash
# Chunk a text file by sentences
python document_chunker.py sample_document.txt sentence

# Chunk a PDF with fixed-size chunks
python document_chunker.py document.pdf fixed --chunk-size 1000

# Save chunks to file
python document_chunker.py sample_document.txt paragraph --output chunks.txt
```

### Advanced Usage
```bash
# Sliding window with 50% overlap
python document_chunker.py sample_document.txt sliding --window-size 1000 --step-size 500

# Section-based chunking with custom heading pattern
python document_chunker.py sample_document.txt section --heading-pattern "^##\\s+"

# Token-based chunking for BERT models
python document_chunker.py sample_document.txt token --max-tokens 512
```

## Output Format

Each chunk includes:
- **text**: The actual chunk content
- **chunk_id**: Unique identifier for the chunk
- **method**: The chunking method used
- **Additional metadata**: Method-specific information (line numbers, character positions, etc.)

### Example Output
```
=== Generated 3 chunks ===

--- Chunk 1 ---
Metadata: {'chunk_id': 0, 'start_sentence': 1, 'end_sentence': 3, 'method': 'sentence_based'}
Content:
Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models. The goal is to learn a mapping from inputs to outputs based on example input-output pairs. Common supervised learning algorithms include linear regression and decision trees.
--------------------------------------------------
```

## File Format Support

- **TXT**: Plain text files
- **PDF**: PDF documents (requires PyPDF2)
- **DOC/DOCX**: Microsoft Word documents (requires python-docx)

## Dependencies

- **Core**: No external dependencies for basic text processing
- **PDF Support**: PyPDF2
- **DOC Support**: python-docx
- **Token Counting**: transformers, torch
- **Advanced NLP**: nltk, spacy (optional)

## Use Cases

This tool is particularly useful for:

1. **RAG System Preparation**: Chunking documents for vector databases
2. **Document Analysis**: Breaking down large documents for analysis
3. **Content Processing**: Preparing text for machine learning pipelines
4. **Search Optimization**: Creating searchable document segments
5. **API Integration**: Preparing documents for embedding APIs

## Tips for RAG Systems

1. **Overlap**: Use sliding window or fixed-size with overlap to maintain context
2. **Token Limits**: Use token-based chunking for models with specific token limits
3. **Semantic Boundaries**: Use sentence or paragraph chunking to preserve meaning
4. **Metadata**: Keep metadata for better retrieval and context understanding
5. **Section Awareness**: Use section-based chunking for structured documents

## Troubleshooting

### Common Issues

1. **Import Errors**: Install missing dependencies with `pip install -r requirements.txt`
2. **PDF Issues**: Ensure PyPDF2 is installed for PDF processing
3. **Token Counting**: Install transformers for accurate token counting
4. **Memory Issues**: For large documents, consider using smaller chunk sizes

### Performance Tips

1. Use appropriate chunk sizes based on your embedding model
2. Consider overlap for better context preservation
3. Use section-based chunking for structured documents
4. Monitor memory usage with very large documents
