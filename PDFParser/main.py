from pdf_parser import PDFParser
import json


def main():
    # Initialize the parser with custom chunk size and overlap
    parser = PDFParser(chunk_size=1000, chunk_overlap=200)

    # Example 1: Parse a PDF and get structured results
    print("Example 1: Parsing PDF to structured data")
    print("-" * 80)

    pdf_path = "sample.pdf"  # Replace with your PDF file path

    try:
        # Parse the PDF
        results = parser.parse_pdf(pdf_path)

        # Display summary
        print(f"Successfully parsed: {pdf_path}")
        print(f"Total chunks created: {len(results)}")
        print(f"\nFirst chunk preview:")
        print(results[0]['text'][:200] + "...")

        # Save results to JSON for RAG ingestion
        with open("chunks.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("\nChunks saved to: chunks.json")

    except FileNotFoundError:
        print(f"Error: PDF file '{pdf_path}' not found.")
        print("Please update the pdf_path variable with a valid PDF file.")
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")

    print("\n" + "=" * 80 + "\n")

    # Example 2: Parse PDF and save to text file
    print("Example 2: Parsing PDF to text file")
    print("-" * 80)

    try:
        parser.parse_pdf_to_file(pdf_path, "output.txt")
        print(f"Chunks saved to: output.txt")
    except FileNotFoundError:
        print(f"Error: PDF file '{pdf_path}' not found.")
    except Exception as e:
        print(f"Error: {str(e)}")

    print("\n" + "=" * 80 + "\n")

    # Example 3: Custom chunking parameters
    print("Example 3: Using custom chunking parameters")
    print("-" * 80)

    # Smaller chunks for more granular retrieval
    small_chunk_parser = PDFParser(chunk_size=500, chunk_overlap=100)

    try:
        small_results = small_chunk_parser.parse_pdf(pdf_path)
        print(f"Chunks with size=500: {len(small_results)}")
    except:
        print("Skipping due to missing PDF file")

    # Larger chunks for more context
    large_chunk_parser = PDFParser(chunk_size=2000, chunk_overlap=400)

    try:
        large_results = large_chunk_parser.parse_pdf(pdf_path)
        print(f"Chunks with size=2000: {len(large_results)}")
    except:
        print("Skipping due to missing PDF file")

if __name__ == "__main__":
    main()