import csv
import json
from pathlib import Path

def parse_csv_for_rag(csv_file_path, output_file_path=None):

    documents = []

    # Read the CSV file
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        
        reader = csv.DictReader(csvfile)

        # Get column names
        fieldnames = reader.fieldnames
        print(f"Found columns: {fieldnames}")

        # Process each row
        for idx, row in enumerate(reader):
            # Create a text representation of the row
            text_parts = []
            for key, value in row.items():
                if value:  # Only include non-empty values
                    text_parts.append(f"{key}: {value}")

            # Combine into a single text document
            text = " | ".join(text_parts)

            # Create a document with text and metadata
            document = {
                "id": f"doc_{idx}",
                "text": text,
                "metadata": {
                    "source": csv_file_path,
                    "row_number": idx,
                    **row  # Include all CSV fields as metadata
                }
            }

            documents.append(document)

    print(f"Processed {len(documents)} documents")

    # Optionally save to JSON file
    if output_file_path:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(documents, outfile, indent=2, ensure_ascii=False)
        print(f"Saved documents to {output_file_path}")

    return documents

def main():
    
    # Input and output file paths
    csv_file = "big_data.csv"
    output_file = "rag_documents.json"

    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found!")
        return

    documents = parse_csv_for_rag(csv_file, output_file)

    if documents:
        print("\nSample document:")
        print(json.dumps(documents[574], indent=2))
        total = len(documents)
        print(f"\nTotal Chunks: {total}")
        
        
if __name__ == "__main__":
    main()