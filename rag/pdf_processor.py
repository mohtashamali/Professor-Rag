"""
PDF Processing Module
Handles PDF loading, text extraction, and chunking
"""
import PyPDF2
from typing import List, Dict
import re

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor
        
        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf(self, pdf_path: str) -> str:
        """
        Load and extract text from a PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error loading PDF {pdf_path}: {str(e)}")
    
    def load_multiple_pdfs(self, pdf_paths: List[str]) -> Dict[str, str]:
        """
        Load multiple PDFs and return a dictionary of filename to text
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            Dictionary mapping filename to extracted text
        """
        pdf_texts = {}
        for pdf_path in pdf_paths:
            filename = pdf_path.split('/')[-1]
            try:
                pdf_texts[filename] = self.load_pdf(pdf_path)
            except Exception as e:
                print(f"Warning: Could not load {filename}: {str(e)}")
        return pdf_texts
    
    def clean_text(self, text: str) -> str:

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep mathematical symbols
        text = re.sub(r'[^\w\s\.\,\:\;\!\?\-\+\=\*\/\(\)\[\]\{\}\<\>\^\~\'\"]', '', text)
        return text.strip()
    
    def split_into_chunks(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into overlapping chunks for better retrieval
        
        Args:
            text: Text to split
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of dictionaries with 'text' and 'metadata' keys
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < text_length:
                # Look for period followed by space or newline
                last_period = text.rfind('. ', start, end)
                if last_period != -1 and last_period > start + self.chunk_size // 2:
                    end = last_period + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_data = {
                    'text': chunk_text,
                    'metadata': metadata or {},
                    'start_index': start,
                    'end_index': end
                }
                chunks.append(chunk_data)
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_pdfs(self, pdf_paths: List[str]) -> List[Dict]:
        """
        Complete pipeline: load PDFs, clean text, and create chunks
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of text chunks with metadata
        """
        all_chunks = []
        pdf_texts = self.load_multiple_pdfs(pdf_paths)
        
        for filename, text in pdf_texts.items():
            cleaned_text = self.clean_text(text)
            chunks = self.split_into_chunks(
                cleaned_text,
                metadata={'source': filename}
            )
            all_chunks.extend(chunks)
        
        return all_chunks