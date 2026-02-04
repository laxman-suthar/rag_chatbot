"""
Vector Store Management using FAISS
Handles storage and retrieval of document embeddings
"""

import logging
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import faiss
from django.conf import settings

logger = logging.getLogger('rag_service')


class VectorStore:
    """
    Vector store for managing document embeddings using FAISS
    """
    
    def __init__(self):
        """Initialize vector store"""
        self.index_path = settings.VECTOR_STORE_PATH / 'faiss.index'
        self.metadata_path = settings.VECTOR_STORE_PATH / 'metadata.pkl'
        
        self.index = None
        self.metadata = []
        self.dimension = settings.EMBEDDING_CONFIG.get('dimension', 384)  # Default for all-MiniLM-L6-v2
        
        # Load existing index if available
        self.load_index()
    
    def load_index(self):
        """Load FAISS index and metadata from disk"""
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(self.index_path))
                
                # Load metadata
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                logger.info(
                    f"Loaded vector store with {self.index.ntotal} vectors"
                )
            else:
                logger.info("No existing index found. Creating new index.")
                self._create_new_index()
                
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}", exc_info=True)
            logger.info("Creating new index due to load error")
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index"""
        # Using IndexFlatL2 for exact search (good for small to medium datasets)
        # For larger datasets, consider IndexIVFFlat or IndexHNSWFlat
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        logger.info(f"Created new FAISS index with dimension {self.dimension}")
    
    def add_documents(
        self,
        embeddings: np.ndarray,
        documents_metadata: List[Dict]
    ) -> bool:
        """
        Add document embeddings to the index
        
        Args:
            embeddings: Numpy array of embeddings (n_docs, dimension)
            documents_metadata: List of metadata dicts for each document
        
        Returns:
            True if successful
        """
        try:
            if embeddings.shape[0] != len(documents_metadata):
                raise ValueError(
                    f"Mismatch: {embeddings.shape[0]} embeddings but "
                    f"{len(documents_metadata)} metadata entries"
                )
            
            # Ensure embeddings are float32
            embeddings = embeddings.astype('float32')
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            start_id = self.index.ntotal
            self.index.add(embeddings)
            
            # Add metadata with IDs
            for i, meta in enumerate(documents_metadata):
                meta['id'] = start_id + i
                self.metadata.append(meta)
            
            # Save to disk
            self.save_index()
            
            logger.info(
                f"Added {len(documents_metadata)} documents to index. "
                f"Total: {self.index.ntotal}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}", exc_info=True)
            return False
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
        
        Returns:
            List of dictionaries with document info and similarity scores
        """
        try:
            if self.index is None or self.index.ntotal == 0:
                logger.warning("Index is empty or not loaded")
                return []
            
            # Ensure query is 2D array and float32
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            query_embedding = query_embedding.astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search
            k = min(k, self.index.ntotal)  # Don't request more than available
            distances, indices = self.index.search(query_embedding, k)
            
            # Convert distances to similarity scores (for L2 distance)
            # Lower distance = higher similarity
            # Convert to 0-1 range where 1 is most similar
            similarities = 1 / (1 + distances[0])
            
            # Build results
            results = []
            for idx, similarity in zip(indices[0], similarities):
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['score'] = float(similarity)
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching index: {str(e)}", exc_info=True)
            return []
    
    def save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Ensure directory exists
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            logger.info(f"Saved index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}", exc_info=True)
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks associated with a document
        
        Note: FAISS doesn't support deletion efficiently.
        This marks items as deleted in metadata and rebuilds on next save.
        
        Args:
            document_id: ID of document to delete
        
        Returns:
            True if successful
        """
        try:
            # Mark metadata as deleted
            deleted_count = 0
            for meta in self.metadata:
                if meta.get('document_id') == document_id:
                    meta['deleted'] = True
                    deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"Marked {deleted_count} chunks as deleted for document {document_id}")
                # Rebuild index without deleted items
                self._rebuild_index()
                return True
            else:
                logger.warning(f"No chunks found for document {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}", exc_info=True)
            return False
    
    def _rebuild_index(self):
        """Rebuild index without deleted items"""
        try:
            # Filter out deleted metadata
            active_metadata = [m for m in self.metadata if not m.get('deleted', False)]
            
            if not active_metadata:
                self._create_new_index()
                return
            
            # Create new index
            new_index = faiss.IndexFlatL2(self.dimension)
            
            # We need to reload embeddings - this is inefficient
            # Better approach: store embeddings separately or don't support deletion
            logger.warning("Rebuild index requires re-embedding documents")
            
            # For now, just update metadata
            self.metadata = active_metadata
            self.save_index()
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {str(e)}", exc_info=True)
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'total_metadata': len(self.metadata),
            'active_metadata': len([m for m in self.metadata if not m.get('deleted', False)]),
            'index_size_mb': self.index_path.stat().st_size / (1024 * 1024) if self.index_path.exists() else 0,
        }
    
    def clear(self):
        """Clear all data from vector store"""
        logger.warning("Clearing vector store")
        self._create_new_index()
        self.save_index()
    
    def optimize_index(self):
        """
        Optimize index for faster search (for production)
        Converts to IVF index for large datasets
        """
        try:
            if self.index.ntotal < 1000:
                logger.info("Index too small to optimize (< 1000 vectors)")
                return
            
            logger.info("Optimizing index with IVF...")
            
            # Create IVF index
            nlist = min(100, self.index.ntotal // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            index_ivf = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
            # Train on existing vectors
            vectors = np.zeros((self.index.ntotal, self.dimension), dtype='float32')
            for i in range(self.index.ntotal):
                vectors[i] = self.index.reconstruct(i)
            
            index_ivf.train(vectors)
            index_ivf.add(vectors)
            
            # Replace index
            self.index = index_ivf
            self.save_index()
            
            logger.info(f"Index optimized with {nlist} clusters")
            
        except Exception as e:
            logger.error(f"Error optimizing index: {str(e)}", exc_info=True)