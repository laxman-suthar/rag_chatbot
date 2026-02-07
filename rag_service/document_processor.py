"""
Document Processor using LangChain for document loading and text splitting
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple
from django.conf import settings

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader
)

logger = logging.getLogger('rag_service')


class DocumentProcessor:
    """
    Process documents using LangChain loaders and splitters
    """
    
    def __init__(self):
        """Initialize document processor with LangChain text splitter"""
        self.chunk_size = settings.RAG_CONFIG['chunk_size']
        self.chunk_overlap = settings.RAG_CONFIG['chunk_overlap']
        
        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info(
            f"LangChain document processor initialized. "
            f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}"
        )
    
    def process_document(
        self,
        file_path: str,
        document_id: str,
        document_title: str
    ) -> Tuple[List[str], List[Dict]]:
        """
        Process a document using LangChain and return chunks with metadata
        
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
            
            logger.info(f"Processing document with LangChain: {document_title} ({file_ext})")
            
            # Load document using appropriate LangChain loader
            loader = self._get_loader(file_path, file_ext)
            documents = loader.load()
            
            if not documents:
                raise ValueError("No content extracted from document")
            
            # Split documents into chunks using LangChain
            chunks = self.text_splitter.split_documents(documents)
            
            # Extract text and create metadata
            chunk_texts = []
            metadata_list = []
            
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.page_content
                chunk_texts.append(chunk_text)
                
                # Combine LangChain metadata with our metadata
                metadata = {
                    'document_id': document_id,
                    'document_title': document_title,
                    'chunk_index': i,
                    'content': chunk_text,
                    'file_type': file_ext[1:],
                    'page_number': chunk.metadata.get('page', chunk.metadata.get('page_number')),
                    'source': chunk.metadata.get('source', str(file_path)),
                }
                metadata_list.append(metadata)
            
            logger.info(
                f"Processed {document_title} with LangChain: {len(chunks)} chunks"
            )
            
            return chunk_texts, metadata_list
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", exc_info=True)
            raise
    
    def _get_loader(self, file_path: Path, file_ext: str):
        """Get appropriate LangChain document loader for file type"""
        loaders = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.html': UnstructuredHTMLLoader,
            '.htm': UnstructuredHTMLLoader,
            '.md': UnstructuredMarkdownLoader,
        }
        
        loader_class = loaders.get(file_ext)
        if not loader_class:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        return loader_class(str(file_path))
    
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
                try:
                    loader = PyPDFLoader(str(file_path))
                    documents = loader.load()
                    stats['page_count'] = len(documents)
                except:
                    pass
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}", exc_info=True)
            return {}


class DocumentIndexer:
    """
    High-level document indexing with LangChain and vector store integration
    """
    
    def __init__(self, embedding_model, vector_store):
        """
        Initialize document indexer
        
        Args:
            embedding_model: EmbeddingModel instance (LangChain-based)
            vector_store: VectorStore instance
        """
        self.processor = DocumentProcessor()
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        
        logger.info("LangChain document indexer initialized")
    
    def index_document(
        self,
        file_path: str,
        document_id: str,
        document_title: str,
        batch_size: int = 32
    ) -> int:
        """
        Index a document into the vector store using LangChain
        
        Args:
            file_path: Path to document file
            document_id: Unique document ID
            document_title: Document title
            batch_size: Batch size for embedding generation
        
        Returns:
            Number of chunks indexed
        """
        try:
            # Process document with LangChain
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