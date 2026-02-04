"""
Embedding Model wrapper using Sentence Transformers
"""

import logging
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger('rag_service')


class EmbeddingModel:
    """
    Wrapper for embedding model with caching support
    """
    
    def __init__(self):
        """Initialize embedding model"""
        self.model_name = settings.EMBEDDING_CONFIG['model_name']
        self.device = settings.EMBEDDING_CONFIG.get('device', 'cpu')
        self.cache_enabled = settings.EMBEDDING_CONFIG.get('cache_enabled', True)
        self.cache_ttl = settings.EMBEDDING_CONFIG.get('cache_ttl', 3600)  # 1 hour
        
        # Load model
        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
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
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            
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
            'all-MiniLM-L6-v2',          # Fast, 384 dim, good quality
            'all-mpnet-base-v2',          # Best quality, 768 dim, slower
            'all-MiniLM-L12-v2',          # Balance, 384 dim
            'paraphrase-multilingual-MiniLM-L12-v2',  # Multilingual
            'msmarco-distilbert-base-v4', # Good for search
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
            'all-MiniLM-L6-v2': {
                'dimension': 384,
                'max_length': 256,
                'speed': 'fast',
                'quality': 'good',
                'multilingual': False,
            },
            'all-mpnet-base-v2': {
                'dimension': 768,
                'max_length': 384,
                'speed': 'medium',
                'quality': 'best',
                'multilingual': False,
            },
            'all-MiniLM-L12-v2': {
                'dimension': 384,
                'max_length': 256,
                'speed': 'fast',
                'quality': 'very good',
                'multilingual': False,
            },
            'paraphrase-multilingual-MiniLM-L12-v2': {
                'dimension': 384,
                'max_length': 128,
                'speed': 'medium',
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