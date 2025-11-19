"""
Simple PDF Parser for RAG ingestion.

This module provides functionality to extract text from PDF files and chunk it
for ingestion into Retrieval-Augmented Generation (RAG) systems.
"""

from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFParser:
    """Parse PDF files and prepare them for RAG ingestion."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PDF parser.

        Args:
            chunk_size: Maximum size of each text chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text as a single string

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            Exception: If there's an error reading the PDF
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)

                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()

        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")

        return text

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks suitable for RAG ingestion.

        Args:
            text: The text to chunk

        Returns:
            List of text chunks
        """
        chunks = self.text_splitter.split_text(text)
        return chunks

    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Parse a PDF and return chunked text with metadata.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of dictionaries containing chunk text and metadata
        """
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)

        # Chunk the text
        chunks = self.chunk_text(text)

        # Create structured output with metadata
        results = []
        for idx, chunk in enumerate(chunks):
            results.append({
                "chunk_id": idx,
                "text": chunk,
                "source": pdf_path,
                "chunk_size": len(chunk),
                "total_chunks": len(chunks)
            })

        return results

    def parse_pdf_to_file(self, pdf_path: str, output_path: str) -> None:
        """
        Parse a PDF and save the chunks to a text file.

        Args:
            pdf_path: Path to the PDF file
            output_path: Path to save the output file
        """
        results = self.parse_pdf(pdf_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"PDF Parser Results\n")
            f.write(f"Source: {pdf_path}\n")
            f.write(f"Total Chunks: {len(results)}\n")
            f.write("=" * 80 + "\n\n")

            for result in results:
                f.write(f"Chunk {result['chunk_id'] + 1}/{result['total_chunks']}\n")
                f.write(f"Size: {result['chunk_size']} characters\n")
                f.write("-" * 80 + "\n")
                f.write(result['text'])
                f.write("\n\n" + "=" * 80 + "\n\n")