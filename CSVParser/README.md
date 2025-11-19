# CSVParser - CSV to RAG Document Converter

A utility for converting CSV files into RAG-ready JSON documents suitable for ingestion into vector databases and retrieval systems.

## Overview

CSVParser transforms structured CSV data into text-based documents optimized for Retrieval-Augmented Generation (RAG) systems. Each CSV row becomes a searchable document with both text content and structured metadata, making tabular data queryable using semantic search.

## Features

- 📊 **CSV to Text Conversion**: Transforms each row into a human-readable text representation
- 🏷️ **Metadata Preservation**: Retains all CSV fields as structured metadata
- 📝 **JSON Output**: Produces RAG-compatible JSON documents
- 🔍 **Search-Ready**: Text format optimized for semantic similarity search
- ⚡ **Simple & Fast**: Pure Python implementation with minimal dependencies

## Use Cases

- **Customer Data Search**: Convert customer databases into searchable knowledge bases
- **Product Catalogs**: Make product information semantically searchable
- **Log Analysis**: Transform structured logs into queryable text
- **Data Export**: Prepare CSV exports for RAG systems like ChromaDB, Pinecone, or Weaviate
- **Knowledge Base**: Convert spreadsheets into LLM-friendly documents

## Requirements

- Python 3.8+
- No external dependencies (uses only standard library)

## Installation

No installation needed beyond Python standard library. Just ensure you have Python 3.8+:

```bash
python --version
```

## Project Structure

```
csvparser/
├── main.py              # CSV parser script
├── sample_data.csv      # Sample dataset (1000 rows)
├── big_data.csv         # Larger dataset (10000 rows)
├── rag_documents.json   # Output JSON file (generated)
└── README.md            # This file
```

## Usage

### Basic Usage

Run the parser on the default file (`big_data.csv`):

```bash
python main.py
```

Output:
```
Found columns: ['id', 'first_name', 'last_name', 'email', 'gender', 'ip_address']
Processed 10000 documents
Saved documents to rag_documents.json

Sample document:
{
  "id": "doc_574",
  "text": "id: 575 | first_name: Jane | last_name: Smith | email: jane@example.com | gender: Female | ip_address: 192.168.1.1",
  "metadata": {
    "source": "big_data.csv",
    "row_number": 574,
    "id": "575",
    "first_name": "Jane",
    "last_name": "Smith",
    "email": "jane@example.com",
    "gender": "Female",
    "ip_address": "192.168.1.1"
  }
}

Total Chunks: 10000
```

### Using Your Own CSV File

Modify `main.py` to point to your CSV file:

```python
def main():
    # Change these paths
    csv_file = "your_data.csv"
    output_file = "output_documents.json"
    
    # ... rest of code
```

Then run:
```bash
python main.py
```

### Programmatic Usage

Use the parser as a module in your own scripts:

```python
from main import parse_csv_for_rag

# Parse CSV to documents
documents = parse_csv_for_rag(
    csv_file_path="data.csv",
    output_file_path="output.json"  # Optional
)

# Access documents
for doc in documents[:5]:
    print(doc["text"])
    print(doc["metadata"])
```

## Output Format

Each CSV row is converted to a JSON document with this structure:

```json
{
  "id": "doc_0",
  "text": "column1: value1 | column2: value2 | column3: value3",
  "metadata": {
    "source": "input.csv",
    "row_number": 0,
    "column1": "value1",
    "column2": "value2",
    "column3": "value3"
  }
}
```

### Fields Explained

- **id**: Unique document identifier (`doc_0`, `doc_1`, etc.)
- **text**: Pipe-separated text representation of the row (used for embedding/search)
- **metadata**: 
  - `source`: Original CSV filename
  - `row_number`: Zero-indexed row position
  - All original CSV columns as key-value pairs

## Sample Data

### sample_data.csv
1000-row dataset with person information:
- id, first_name, last_name, email, ip_address, favorite_animal

### big_data.csv  
10,000-row dataset with person information:
- id, first_name, last_name, email, gender, ip_address

## Integration with RAG Systems

### ChromaDB Example

```python
import json
import chromadb

# Load parsed documents
with open("rag_documents.json", "r") as f:
    documents = json.load(f)

# Create ChromaDB collection
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("csv_data")

# Add documents
collection.add(
    ids=[doc["id"] for doc in documents],
    documents=[doc["text"] for doc in documents],
    metadatas=[doc["metadata"] for doc in documents]
)

# Query
results = collection.query(
    query_texts=["Find someone with email domain gmail.com"],
    n_results=5
)
```

### LangChain Example

```python
import json
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Load parsed documents
with open("rag_documents.json", "r") as f:
    raw_docs = json.load(f)

# Convert to LangChain documents
documents = [
    Document(
        page_content=doc["text"],
        metadata=doc["metadata"]
    )
    for doc in raw_docs
]

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Search
results = vectorstore.similarity_search("john smith", k=5)
```

## Customization

### Modify Text Formatting

Change how rows are converted to text in the `parse_csv_for_rag()` function:

```python
# Current format: "column1: value1 | column2: value2"
text = " | ".join(text_parts)

# Alternative: Sentence format
text = ". ".join([f"{key} is {value}" for key, value in row.items()])

# Alternative: Natural language
text = f"Person {row['first_name']} {row['last_name']} with email {row['email']}"
```

### Filter Columns

Only include specific columns in the text:

```python
# Only include specific columns
important_columns = ["first_name", "last_name", "email"]
for key, value in row.items():
    if key in important_columns and value:
        text_parts.append(f"{key}: {value}")
```

### Skip Empty Rows

```python
# Skip rows with missing critical data
if not row.get("email") or not row.get("first_name"):
    continue
```

## Advanced Features

### Handling Large CSV Files

For very large CSV files, process in batches:

```python
import csv
import json

def parse_csv_in_batches(csv_file, batch_size=1000):
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        batch = []
        
        for idx, row in enumerate(reader):
            # Process row...
            document = {...}
            batch.append(document)
            
            if len(batch) >= batch_size:
                # Save or process batch
                yield batch
                batch = []
        
        if batch:
            yield batch
```

### Adding Custom Metadata

Enrich documents with computed fields:

```python
document = {
    "id": f"doc_{idx}",
    "text": text,
    "metadata": {
        "source": csv_file_path,
        "row_number": idx,
        "processed_date": datetime.now().isoformat(),
        "text_length": len(text),
        "domain": row["email"].split("@")[1] if "@" in row.get("email", "") else None,
        **row
    }
}
```

### Combining Multiple CSV Files

```python
import glob

all_documents = []
for csv_file in glob.glob("*.csv"):
    docs = parse_csv_for_rag(csv_file, output_file_path=None)
    all_documents.extend(docs)

# Save combined output
with open("all_documents.json", "w") as f:
    json.dump(all_documents, f, indent=2)
```

## Performance Tips

- **Large Files**: Use batch processing for files with >100K rows
- **Memory**: Process and stream documents instead of loading all in memory
- **Encoding**: Specify encoding if you have special characters: `encoding='utf-8'`
- **Empty Values**: The parser already skips empty values to reduce noise

## Troubleshooting

**Issue**: "File not found"
- Ensure CSV file exists in the same directory
- Use absolute paths if running from different directory

**Issue**: Encoding errors
- Add encoding parameter: `open(file, 'r', encoding='utf-8')`
- Try `encoding='latin-1'` or `encoding='iso-8859-1'` for legacy data

**Issue**: Empty output
- Check CSV file has headers and data rows
- Verify CSV format is valid (no mismatched quotes)

**Issue**: Memory error with large files
- Implement batch processing (see Advanced Features)
- Process to vector DB directly without intermediate JSON

## Output File Size

The JSON output is typically 3-5x larger than the CSV input due to:
- JSON formatting and structure
- Duplicate data in both text and metadata fields
- Pretty printing with indentation

For a 10,000-row CSV (~1 MB), expect ~3-5 MB JSON output.

## Next Steps

- Integrate with your RAG system (ChromaDB, Pinecone, Weaviate)
- Add embedding generation before storing
- Implement incremental updates for changed rows
- Add data validation and cleaning
- Create a CLI with argparse for flexible usage
- Add support for Excel, TSV, and other formats

## Related Projects

See other demos in this repository:
- **ChromaDemo**: Vector database with ChromaDB
- **BookSearch**: Full RAG implementation with Ollama
- **FAISSDemo**: FAISS vector search

## License

Educational demo project.
