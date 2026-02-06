"""
Vector Store Management using ChromaDB
Handles storage and retrieval of document embeddings
"""

import logging
import numpy as np
from typing import List, Dict
import chromadb
from chromadb.config import Settings as ChromaSettings
from django.conf import settings

logger = logging.getLogger('rag_service')


class VectorStore:
    """
    Vector store for managing document embeddings using ChromaDB
    """

    def __init__(self):
        """Initialize vector store"""
        self.persist_path = settings.VECTOR_STORE_PATH / 'chroma'
        self.collection_name = "documents"
        self.client = None
        self.collection = None

        self._init_client()

    def _init_client(self):
        """Initialize Chroma client and collection"""
        try:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            telemetry_enabled = settings.EMBEDDING_CONFIG.get('chroma_telemetry', False)
            chroma_settings = ChromaSettings(anonymized_telemetry=telemetry_enabled)
            self.client = chromadb.PersistentClient(
                path=str(self.persist_path),
                settings=chroma_settings
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB vector store initialized")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}", exc_info=True)
            raise

    def add_documents(
        self,
        embeddings: np.ndarray,
        documents_metadata: List[Dict]
    ) -> bool:
        """
        Add document embeddings to the vector store
        """
        try:
            if embeddings.shape[0] != len(documents_metadata):
                raise ValueError(
                    f"Mismatch: {embeddings.shape[0]} embeddings but "
                    f"{len(documents_metadata)} metadata entries"
                )

            # Ensure embeddings are float32
            embeddings = embeddings.astype('float32')

            ids = []
            metadatas = []
            documents = []

            for meta in documents_metadata:
                document_id = str(meta.get('document_id', 'unknown'))
                chunk_index = str(meta.get('chunk_index', '0'))
                chunk_id = f"{document_id}:{chunk_index}"
                ids.append(chunk_id)

                # Extract and remove content from metadata
                content = meta.get('content', '')
                documents.append(content)

                # Remove None values (Chroma rejects them)
                clean_meta = {
                    k: v for k, v in meta.items()
                    if v is not None and k != 'content'
                }
                clean_meta['document_id'] = document_id
                metadatas.append(clean_meta)

            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas
            )

            logger.info(f"Added {len(documents_metadata)} documents to ChromaDB")
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
        """
        try:
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            query_embedding = query_embedding.astype('float32')

            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=k,
                include=["metadatas", "documents", "distances"]  # ← REMOVE "ids"
            )

            if not results or not results.get('ids'):  # ids is still returned by default
                return []

            hits = []
            ids = results['ids'][0]
            metadatas = results.get('metadatas', [[]])[0]
            documents = results.get('documents', [[]])[0]
            distances = results.get('distances', [[]])[0]

            for i in range(len(ids)):
                meta = (metadatas[i] or {}).copy()
                meta['id'] = ids[i]
                meta['content'] = documents[i] if i < len(documents) else ""

                # Chroma cosine distance: similarity = 1 - distance
                distance = distances[i] if i < len(distances) else 1.0
                meta['score'] = float(max(0.0, 1.0 - distance))

                hits.append(meta)

            return hits

        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}", exc_info=True)
            return []
        
    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks associated with a document
        """
        try:
            doc_id = str(document_id)
            results = self.collection.get(
                where={"document_id": doc_id},
                include=["ids"]
            )
            ids = results.get("ids", [])
            if not ids:
                logger.warning(f"No chunks found for document {document_id}")
                return False

            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunks for document {document_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}", exc_info=True)
            return False

    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        try:
            total = self.collection.count()
        except Exception:
            total = 0
        return {
            'total_vectors': total,
            'dimension': settings.EMBEDDING_CONFIG.get('dimension', 'unknown'),
            'total_metadata': total,
            'active_metadata': total,
            'index_size_mb': 0,
        }

    def clear(self):
        """Clear all data from vector store"""
        logger.warning("Clearing vector store")
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def optimize_index(self):
        """No-op for ChromaDB (managed internally)"""
        logger.info("ChromaDB manages index optimization automatically")
