#!/usr/bin/env python3
"""
Document Chunker for RAG Systems
A comprehensive tool for chunking documents using various strategies.
"""

import argparse
import os
import re
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import PyPDF2
    import docx
    from transformers import AutoTokenizer
    HAS_ADVANCED_DEPS = True
except ImportError:
    HAS_ADVANCED_DEPS = False


class DocumentChunker:
    """A comprehensive document chunker supporting multiple chunking strategies."""
    
    def __init__(self):
        self.tokenizer = None
        if HAS_ADVANCED_DEPS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            except:
                pass
    
    def read_file(self, file_path: str) -> str:
        """Read content from various file formats."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif suffix == '.pdf':
            if not HAS_ADVANCED_DEPS:
                raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
            return self._read_pdf(file_path)
        
        elif suffix in ['.doc', '.docx']:
            if not HAS_ADVANCED_DEPS:
                raise ImportError("python-docx is required for DOC processing. Install with: pip install python-docx")
            return self._read_docx(file_path)
        
        else:
            # Try to read as text file
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _read_pdf(self, file_path: Path) -> str:
        """Read PDF file content."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _read_docx(self, file_path: Path) -> str:
        """Read DOCX file content."""
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def chunk_by_lines(self, text: str, max_lines: int = 10) -> List[Dict[str, Any]]:
        """Chunk text by lines."""
        lines = text.split('\n')
        chunks = []
        
        for i in range(0, len(lines), max_lines):
            chunk_lines = lines[i:i + max_lines]
            chunk_text = '\n'.join(chunk_lines)
            chunks.append({
                'text': chunk_text,
                'chunk_id': len(chunks),
                'start_line': i + 1,
                'end_line': min(i + max_lines, len(lines)),
                'method': 'line_by_line'
            })
        
        return chunks
    
    def chunk_fixed_size(self, text: str, chunk_size: int = 1000, overlap: int = 0) -> List[Dict[str, Any]]:
        """Chunk text into fixed-size chunks."""
        # Validate overlap is less than chunk_size to prevent infinite loops
        if overlap >= chunk_size:
            raise ValueError(f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})")
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunks.append({
                'text': chunk_text,
                'chunk_id': len(chunks),
                'start_char': start,
                'end_char': end,
                'method': 'fixed_size'
            })
            
            # Calculate step size (always advance by at least 1 to prevent hanging)
            step = max(1, chunk_size - overlap) if overlap > 0 else chunk_size
            start += step
        
        return chunks
    
    def chunk_sliding_window(self, text: str, window_size: int = 1000, step_size: int = 500) -> List[Dict[str, Any]]:
        """Chunk text using sliding window approach."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + window_size, len(text))
            chunk_text = text[start:end]
            
            chunks.append({
                'text': chunk_text,
                'chunk_id': len(chunks),
                'start_char': start,
                'end_char': end,
                'method': 'sliding_window'
            })
            
            start += step_size
        
        return chunks
    
    def chunk_by_sentences(self, text: str, max_sentences: int = 5) -> List[Dict[str, Any]]:
        """Chunk text by sentences."""
        # Simple sentence splitting (can be improved with NLTK/spaCy)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), max_sentences):
            chunk_sentences = sentences[i:i + max_sentences]
            chunk_text = '. '.join(chunk_sentences)
            if chunk_text and not chunk_text.endswith('.'):
                chunk_text += '.'
            
            chunks.append({
                'text': chunk_text,
                'chunk_id': len(chunks),
                'start_sentence': i + 1,
                'end_sentence': min(i + max_sentences, len(sentences)),
                'method': 'sentence_based'
            })
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str, max_paragraphs: int = 3) -> List[Dict[str, Any]]:
        """Chunk text by paragraphs."""
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        for i in range(0, len(paragraphs), max_paragraphs):
            chunk_paragraphs = paragraphs[i:i + max_paragraphs]
            chunk_text = '\n\n'.join(chunk_paragraphs)
            
            chunks.append({
                'text': chunk_text,
                'chunk_id': len(chunks),
                'start_paragraph': i + 1,
                'end_paragraph': min(i + max_paragraphs, len(paragraphs)),
                'method': 'paragraph_based'
            })
        
        return chunks
    
    def chunk_by_pages(self, text: str, lines_per_page: int = 50) -> List[Dict[str, Any]]:
        """Chunk text by pages (simulated)."""
        lines = text.split('\n')
        chunks = []
        
        for i in range(0, len(lines), lines_per_page):
            page_lines = lines[i:i + lines_per_page]
            chunk_text = '\n'.join(page_lines)
            
            chunks.append({
                'text': chunk_text,
                'chunk_id': len(chunks),
                'page_number': (i // lines_per_page) + 1,
                'start_line': i + 1,
                'end_line': min(i + lines_per_page, len(lines)),
                'method': 'page_based'
            })
        
        return chunks
    
    def chunk_by_sections(self, text: str, heading_pattern: str = r'^#{1,6}\s+') -> List[Dict[str, Any]]:
        """Chunk text by sections/headings."""
        lines = text.split('\n')
        chunks = []
        current_section = []
        current_heading = "Introduction"
        
        for line in lines:
            if re.match(heading_pattern, line.strip()):
                # Save previous section
                if current_section:
                    chunk_text = '\n'.join(current_section)
                    chunks.append({
                        'text': chunk_text,
                        'chunk_id': len(chunks),
                        'section_heading': current_heading,
                        'method': 'section_based'
                    })
                
                # Start new section
                current_heading = line.strip()
                current_section = [line]
            else:
                current_section.append(line)
        
        # Add the last section
        if current_section:
            chunk_text = '\n'.join(current_section)
            chunks.append({
                'text': chunk_text,
                'chunk_id': len(chunks),
                'section_heading': current_heading,
                'method': 'section_based'
            })
        
        return chunks
    
    def chunk_by_tokens(self, text: str, max_tokens: int = 512) -> List[Dict[str, Any]]:
        """Chunk text by token count."""
        if not self.tokenizer:
            # Fallback to character-based approximation (4 chars ≈ 1 token)
            return self.chunk_fixed_size(text, chunk_size=max_tokens * 4)
        
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append({
                'text': chunk_text,
                'chunk_id': len(chunks),
                'start_token': i,
                'end_token': min(i + max_tokens, len(tokens)),
                'token_count': len(chunk_tokens),
                'method': 'token_based'
            })
        
        return chunks
    
    def chunk_document(self, file_path: str, method: str, **kwargs) -> List[Dict[str, Any]]:
        """Main method to chunk a document using specified method."""
        text = self.read_file(file_path)
        
        method_map = {
            'line': self.chunk_by_lines,
            'fixed': self.chunk_fixed_size,
            'sliding': self.chunk_sliding_window,
            'sentence': self.chunk_by_sentences,
            'paragraph': self.chunk_by_paragraphs,
            'page': self.chunk_by_pages,
            'section': self.chunk_by_sections,
            'token': self.chunk_by_tokens
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown chunking method: {method}")
        
        return method_map[method](text, **kwargs)


def print_chunks(chunks: List[Dict[str, Any]], show_metadata: bool = True):
    """Print chunks to terminal with optional metadata."""
    print(f"\n=== Generated {len(chunks)} chunks ===\n")
    
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1} ---")
        if show_metadata:
            metadata = {k: v for k, v in chunk.items() if k != 'text'}
            print(f"Metadata: {metadata}")
        print(f"Content:\n{chunk['text']}")
        print("-" * 50)


def main():
    """Command-line interface for the document chunker."""
    parser = argparse.ArgumentParser(description="Document Chunker for RAG Systems")
    
    parser.add_argument('file', help='Path to the document file')
    parser.add_argument('method', choices=[
        'line', 'fixed', 'sliding', 'sentence', 
        'paragraph', 'page', 'section', 'token'
    ], help='Chunking method to use')
    
    # Method-specific parameters
    parser.add_argument('--max-lines', type=int, default=10, help='Max lines per chunk (line method)')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size in characters (fixed method)')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap between chunks (fixed method)')
    parser.add_argument('--window-size', type=int, default=1000, help='Window size (sliding method)')
    parser.add_argument('--step-size', type=int, default=500, help='Step size (sliding method)')
    parser.add_argument('--max-sentences', type=int, default=5, help='Max sentences per chunk (sentence method)')
    parser.add_argument('--max-paragraphs', type=int, default=3, help='Max paragraphs per chunk (paragraph method)')
    parser.add_argument('--lines-per-page', type=int, default=50, help='Lines per page (page method)')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max tokens per chunk (token method)')
    parser.add_argument('--heading-pattern', default=r'^#{1,6}\s+', help='Heading pattern regex (section method)')
    
    parser.add_argument('--no-metadata', action='store_true', help='Hide metadata in output')
    parser.add_argument('--output', help='Output file path (optional)')
    
    args = parser.parse_args()
    
    try:
        chunker = DocumentChunker()
        
        # Prepare method arguments
        method_args = {}
        if args.method == 'line':
            method_args['max_lines'] = args.max_lines
        elif args.method == 'fixed':
            method_args['chunk_size'] = args.chunk_size
            method_args['overlap'] = args.overlap
        elif args.method == 'sliding':
            method_args['window_size'] = args.window_size
            method_args['step_size'] = args.step_size
        elif args.method == 'sentence':
            method_args['max_sentences'] = args.max_sentences
        elif args.method == 'paragraph':
            method_args['max_paragraphs'] = args.max_paragraphs
        elif args.method == 'page':
            method_args['lines_per_page'] = args.lines_per_page
        elif args.method == 'section':
            method_args['heading_pattern'] = args.heading_pattern
        elif args.method == 'token':
            method_args['max_tokens'] = args.max_tokens
        
        # Chunk the document
        chunks = chunker.chunk_document(args.file, args.method, **method_args)
        
        # Output results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                for i, chunk in enumerate(chunks):
                    f.write(f"=== Chunk {i + 1} ===\n")
                    if not args.no_metadata:
                        metadata = {k: v for k, v in chunk.items() if k != 'text'}
                        f.write(f"Metadata: {metadata}\n")
                    f.write(f"{chunk['text']}\n\n")
            print(f"Chunks saved to {args.output}")
        else:
            print_chunks(chunks, show_metadata=not args.no_metadata)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
