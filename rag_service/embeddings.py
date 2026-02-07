"""
Embedding Model wrapper using LangChain + Gemini embeddings
"""

import logging
import numpy as np
from typing import List, Union
from django.conf import settings
from django.core.cache import cache

# LangChain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger('rag_service')


class EmbeddingModel:
    """
    LangChain-based wrapper for embedding model with caching support
    """
    
    def __init__(self):
        """Initialize embedding model using LangChain"""
        self.provider = settings.EMBEDDING_CONFIG.get('provider', 'gemini')
        self.model_name = settings.EMBEDDING_CONFIG['model_name']
        self.cache_enabled = settings.EMBEDDING_CONFIG.get('cache_enabled', True)
        self.cache_ttl = settings.EMBEDDING_CONFIG.get('cache_ttl', 3600)
        
        if self.provider != 'gemini':
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

        # Initialize LangChain Gemini embeddings
        logger.info(f"Initializing LangChain Gemini embeddings: {self.model_name}")
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                google_api_key=settings.GEMINI_API_KEY
            )
            
            # Get dimension by testing
            self.dimension = self._get_embedding_dimension()
            logger.info(f"LangChain Gemini embeddings ready. Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Error initializing LangChain embeddings: {str(e)}", exc_info=True)
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for text(s) using LangChain
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length
        
        Returns:
            Numpy array of embeddings
        """
        try:
            # Convert single text to list
            single_input = isinstance(texts, str)
            if single_input:
                texts = [texts]
            
            # Check cache for single queries
            if single_input and self.cache_enabled:
                cache_key = self._get_cache_key(texts[0])
                cached_embedding = cache.get(cache_key)
                if cached_embedding is not None:
                    logger.debug("Using cached embedding")
                    return cached_embedding
            
            # Generate embeddings using LangChain
            if show_progress:
                logger.info(f"Generating embeddings for {len(texts)} texts...")
            
            # LangChain's embed_documents handles batching internally
            embeddings_list = self.embeddings.embed_documents(texts)
            embeddings = np.array(embeddings_list, dtype='float32')

            if normalize:
                embeddings = self._normalize_embeddings(embeddings)
            
            # Cache single query embeddings
            if single_input and self.cache_enabled:
                cache_key = self._get_cache_key(texts[0])
                cache.set(cache_key, embeddings, self.cache_ttl)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            raise
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode large batch of texts efficiently using LangChain
        
        Args:
            texts: List of texts
            batch_size: Batch size
            show_progress: Show progress bar
        
        Returns:
            Numpy array of embeddings
        """
        return self.encode(
            texts,
            batch_size=batch_size,
            show_progress=show_progress
        )

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embeddings"""
        if embeddings.ndim == 1:
            denom = np.linalg.norm(embeddings) or 1.0
            return embeddings / denom
        denom = np.linalg.norm(embeddings, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return embeddings / denom

    def _get_embedding_dimension(self) -> int:
        """Determine embedding dimension by embedding a short probe"""
        try:
            probe = self.embeddings.embed_query("test")
            return int(len(probe))
        except Exception as e:
            logger.warning(f"Could not determine dimension: {e}")
            return 768  # Default fallback
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embedding:{self.model_name}:{text_hash}"
    
    def get_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score (0-1)
        """
        try:
            embeddings = self.encode([text1, text2])
            
            # Cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1])
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}", exc_info=True)
            return 0.0
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension
    
    def clear_cache(self):
        """Clear all cached embeddings"""
        if self.cache_enabled:
            logger.info("Clearing embedding cache")
            try:
                cache.delete_pattern("embedding:*")
            except:
                logger.warning("Cache pattern deletion not supported, skipping")