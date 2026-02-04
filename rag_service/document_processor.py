"""
Document Processor for ingesting and chunking documents
Supports PDF, DOCX, TXT, HTML, and Markdown
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple
from django.conf import settings
import numpy as np

# Document processing imports
import PyPDF2
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
import markdown

logger = logging.getLogger('rag_service')


class DocumentProcessor:
    """
    Process documents and prepare them for indexing
    """
    
    def __init__(self):
        """Initialize document processor"""
        self.chunk_size = settings.RAG_CONFIG['chunk_size']
        self.chunk_overlap = settings.RAG_CONFIG['chunk_overlap']
        
        logger.info(
            f"Document processor initialized. "
            f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}"
        )
    
    def process_document(
        self,
        file_path: str,
        document_id: str,
        document_title: str
    ) -> Tuple[List[str], List[Dict]]:
        """
        Process a document and return chunks with metadata
        
        Args:
            file_path: Path to document file
            document_id: Unique document ID
            document_title: Document title
        
        Returns:
            Tuple of (chunks, metadata_list)
        """
        try:
            file_path = Path(file_path)
            file_ext = file_path.suffix.lower()
            
            logger.info(f"Processing document: {document_title} ({file_ext})")
            
            # Extract text based on file type
            if file_ext == '.pdf':
                text, page_map = self._extract_pdf(file_path)
            elif file_ext == '.docx':
                text, page_map = self._extract_docx(file_path)
            elif file_ext == '.txt':
                text, page_map = self._extract_txt(file_path)
            elif file_ext in ['.html', '.htm']:
                text, page_map = self._extract_html(file_path)
            elif file_ext == '.md':
                text, page_map = self._extract_markdown(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            if not text.strip():
                raise ValueError("No text extracted from document")
            
            # Clean text
            text = self._clean_text(text)
            
            # Create chunks
            chunks = self._create_chunks(text)
            
            # Create metadata for each chunk
            metadata_list = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    'document_id': document_id,
                    'document_title': document_title,
                    'chunk_index': i,
                    'content': chunk,
                    'file_type': file_ext[1:],  # Remove dot
                    'page_number': self._get_page_for_chunk(i, page_map) if page_map else None,
                }
                metadata_list.append(metadata)
            
            logger.info(
                f"Processed {document_title}: {len(text)} chars -> {len(chunks)} chunks"
            )
            
            return chunks, metadata_list
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", exc_info=True)
            raise
    
    def _extract_pdf(self, file_path: Path) -> Tuple[str, Dict]:
        """Extract text from PDF"""
        try:
            text_parts = []
            page_map = {}  # Maps chunk index to page number
            current_char_count = 0
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        start_char = current_char_count
                        text_parts.append(page_text)
                        current_char_count += len(page_text)
                        page_map[page_num] = (start_char, current_char_count)
            
            return '\n\n'.join(text_parts), page_map
            
        except Exception as e:
            logger.error(f"Error extracting PDF: {str(e)}", exc_info=True)
            raise
    
    def _extract_docx(self, file_path: Path) -> Tuple[str, Dict]:
        """Extract text from DOCX"""
        try:
            doc = DocxDocument(file_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return '\n\n'.join(paragraphs), {}
            
        except Exception as e:
            logger.error(f"Error extracting DOCX: {str(e)}", exc_info=True)
            raise
    
    def _extract_txt(self, file_path: Path) -> Tuple[str, Dict]:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read(), {}
                
        except UnicodeDecodeError:
            # Try different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read(), {}
    
    def _extract_html(self, file_path: Path) -> Tuple[str, Dict]:
        """Extract text from HTML"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                
                # Remove script and style elements
                for script in soup(['script', 'style']):
                    script.decompose()
                
                # Get text
                text = soup.get_text()
                
                return text, {}
                
        except Exception as e:
            logger.error(f"Error extracting HTML: {str(e)}", exc_info=True)
            raise
    
    def _extract_markdown(self, file_path: Path) -> Tuple[str, Dict]:
        """Extract text from Markdown"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_text = file.read()
                
                # Convert markdown to HTML then to text
                html = markdown.markdown(md_text)
                soup = BeautifulSoup(html, 'html.parser')
                text = soup.get_text()
                
                return text, {}
                
        except Exception as e:
            logger.error(f"Error extracting Markdown: {str(e)}", exc_info=True)
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def _create_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Text to split
        
        Returns:
            List of text chunks
        """
        # Split by sentences first for better chunk boundaries
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                # Keep last few sentences for overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Text to split
        
        Returns:
            List of sentences
        """
        # Simple sentence splitting (could use nltk for better results)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_page_for_chunk(self, chunk_index: int, page_map: Dict) -> int:
        """Estimate page number for a chunk (for PDFs)"""
        # This is a simplified version
        # In production, you'd need more sophisticated mapping
        if not page_map:
            return None
        
        # Return middle page as approximation
        total_pages = len(page_map)
        estimated_page = min(chunk_index + 1, total_pages)
        return estimated_page
    
    def get_document_stats(self, file_path: str) -> Dict:
        """
        Get statistics about a document without full processing
        
        Args:
            file_path: Path to document
        
        Returns:
            Dictionary with document stats
        """
        try:
            file_path = Path(file_path)
            file_ext = file_path.suffix.lower()
            
            stats = {
                'file_name': file_path.name,
                'file_type': file_ext[1:],
                'file_size': file_path.stat().st_size,
            }
            
            # Get page count for PDFs
            if file_ext == '.pdf':
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    stats['page_count'] = len(pdf_reader.pages)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}", exc_info=True)
            return {}


class DocumentIndexer:
    """
    High-level document indexing with vector store integration
    """
    
    def __init__(self, embedding_model, vector_store):
        """
        Initialize document indexer
        
        Args:
            embedding_model: EmbeddingModel instance
            vector_store: VectorStore instance
        """
        self.processor = DocumentProcessor()
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        
        logger.info("Document indexer initialized")
    
    def index_document(
        self,
        file_path: str,
        document_id: str,
        document_title: str,
        batch_size: int = 32
    ) -> int:
        """
        Index a document into the vector store
        
        Args:
            file_path: Path to document file
            document_id: Unique document ID
            document_title: Document title
            batch_size: Batch size for embedding generation
        
        Returns:
            Number of chunks indexed
        """
        try:
            # Process document
            chunks, metadata_list = self.processor.process_document(
                file_path,
                document_id,
                document_title
            )
            
            if not chunks:
                logger.warning(f"No chunks created for document: {document_title}")
                return 0
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = self.embedding_model.encode_batch(
                chunks,
                batch_size=batch_size,
                show_progress=True
            )
            
            # Add to vector store
            success = self.vector_store.add_documents(embeddings, metadata_list)
            
            if success:
                logger.info(f"Successfully indexed {len(chunks)} chunks for {document_title}")
                return len(chunks)
            else:
                logger.error(f"Failed to index document: {document_title}")
                return 0
                
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}", exc_info=True)
            raise
    
    def index_documents_batch(
        self,
        documents: List[Dict],
        batch_size: int = 32
    ) -> Dict:
        """
        Index multiple documents
        
        Args:
            documents: List of dicts with file_path, document_id, document_title
            batch_size: Batch size
        
        Returns:
            Dictionary with indexing results
        """
        results = {
            'success': [],
            'failed': [],
            'total_chunks': 0
        }
        
        for doc in documents:
            try:
                chunk_count = self.index_document(
                    doc['file_path'],
                    doc['document_id'],
                    doc['document_title'],
                    batch_size=batch_size
                )
                
                results['success'].append({
                    'document_id': doc['document_id'],
                    'document_title': doc['document_title'],
                    'chunks': chunk_count
                })
                results['total_chunks'] += chunk_count
                
            except Exception as e:
                logger.error(f"Failed to index {doc['document_title']}: {str(e)}")
                results['failed'].append({
                    'document_id': doc['document_id'],
                    'document_title': doc['document_title'],
                    'error': str(e)
                })
        
        logger.info(
            f"Batch indexing complete. "
            f"Success: {len(results['success'])}, "
            f"Failed: {len(results['failed'])}, "
            f"Total chunks: {results['total_chunks']}"
        )
        
        return results