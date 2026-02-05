"""
Embedding Model wrapper using Gemini embeddings
"""

import logging
import numpy as np
from typing import List, Union
from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger('rag_service')


class EmbeddingModel:
    """
    Wrapper for embedding model with caching support
    """
    
    def __init__(self):
        """Initialize embedding model"""
        self.provider = settings.EMBEDDING_CONFIG.get('provider', 'gemini')
        self.model_name = settings.EMBEDDING_CONFIG['model_name']
        self.device = settings.EMBEDDING_CONFIG.get('device', 'cpu')
        self.cache_enabled = settings.EMBEDDING_CONFIG.get('cache_enabled', True)
        self.cache_ttl = settings.EMBEDDING_CONFIG.get('cache_ttl', 3600)  # 1 hour
        
        if self.provider != 'gemini':
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

        # Initialize Gemini client
        logger.info(f"Initializing Gemini embeddings: {self.model_name}")
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.client = genai
            self.dimension = settings.EMBEDDING_CONFIG.get('dimension')
            if self.dimension:
                logger.info(f"Gemini embeddings ready. Dimension: {self.dimension}")
            else:
                logger.info("Gemini embeddings ready. Dimension: unknown (lazy)")
        except Exception as e:
            logger.error(f"Error initializing Gemini embeddings: {str(e)}", exc_info=True)
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for text(s)
        
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
            
            # Generate embeddings (Gemini API)
            if single_input:
                embeddings = self._embed_single(texts[0])
            else:
                embeddings = self._embed_batch(texts)

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
        Encode large batch of texts efficiently
        
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

    def _embed_single(self, text: str) -> np.ndarray:
        """Embed a single text using Gemini"""
        result = self.client.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_document"
        )
        if hasattr(result, "embedding"):
            embedding = result.embedding
        else:
            embedding = result.get('embedding') or result.get('embeddings')
        return np.array(embedding, dtype='float32')

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts using Gemini (sequential calls)"""
        embeddings = []
        for text in texts:
            embedding = self._embed_single(text)
            embeddings.append(embedding)
        return np.vstack(embeddings)

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
        probe = self._embed_single("dimension probe")
        return int(probe.shape[0])
    
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
            cache.delete_pattern("embedding:*")
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get list of recommended embedding models
        
        Returns:
            List of model names
        """
        return [
            'models/embedding-001',
        ]
    
    @staticmethod
    def get_model_info(model_name: str) -> dict:
        """
        Get information about a model
        
        Args:
            model_name: Model name
        
        Returns:
            Dictionary with model info
        """
        model_specs = {
            'models/embedding-001': {
                'dimension': 'variable',
                'max_length': 'provider-defined',
                'speed': 'fast',
                'quality': 'good',
                'multilingual': True,
            },
        }
        
        return model_specs.get(
            model_name,
            {'dimension': 'unknown', 'info': 'No info available'}
        )


class EmbeddingCache:
    """
    Advanced caching for embeddings with persistence
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize embedding cache
        
        Args:
            cache_dir: Directory to store cache files
        """
        from pathlib import Path
        
        if cache_dir is None:
            cache_dir = settings.VECTOR_STORE_PATH / 'embedding_cache'
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Embedding cache initialized at {self.cache_dir}")
    
    def get(self, key: str) -> np.ndarray:
        """Get cached embedding"""
        import hashlib
        key_hash = hashlib.md5(key.encode()).hexdigest()
        cache_file = self.cache_dir / f"{key_hash}.npy"
        
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception as e:
                logger.error(f"Error loading cache: {str(e)}")
                return None
        return None
    
    def set(self, key: str, embedding: np.ndarray):
        """Cache embedding"""
        import hashlib
        key_hash = hashlib.md5(key.encode()).hexdigest()
        cache_file = self.cache_dir / f"{key_hash}.npy"
        
        try:
            np.save(cache_file, embedding)
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    def clear(self):
        """Clear all cached embeddings"""
        import shutil
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Embedding cache cleared")
