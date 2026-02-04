"""
Management command to index documents from knowledge base
Usage: python manage.py index_documents
"""

import logging
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings
from chatbot.models import Document
from rag_service.embeddings import EmbeddingModel
from rag_service.vector_store import VectorStore
from rag_service.document_processor import DocumentIndexer

logger = logging.getLogger('chatbot')


class Command(BaseCommand):
    help = 'Index documents from knowledge base into vector store'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--path',
            type=str,
            help='Path to knowledge base directory',
            default=None
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing index before indexing'
        )
        parser.add_argument(
            '--reindex',
            action='store_true',
            help='Reindex all existing documents in database'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=32,
            help='Batch size for embedding generation'
        )
    
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting document indexing...'))
        
        # Initialize components
        embedding_model = EmbeddingModel()
        vector_store = VectorStore()
        indexer = DocumentIndexer(embedding_model, vector_store)
        
        # Clear index if requested
        if options['clear']:
            self.stdout.write(self.style.WARNING('Clearing existing index...'))
            vector_store.clear()
            self.stdout.write(self.style.SUCCESS('Index cleared'))
        
        # Reindex existing documents from database
        if options['reindex']:
            self.stdout.write('Reindexing documents from database...')
            self._reindex_from_db(indexer, options['batch_size'])
            return
        
        # Get knowledge base path
        kb_path = options['path'] or settings.KNOWLEDGE_BASE_PATH
        kb_path = Path(kb_path)
        
        if not kb_path.exists():
            self.stdout.write(
                self.style.ERROR(f'Knowledge base path does not exist: {kb_path}')
            )
            return
        
        # Find all documents
        supported_extensions = settings.ALLOWED_EXTENSIONS
        documents_to_index = []
        
        for ext in supported_extensions:
            documents_to_index.extend(kb_path.glob(f'**/*.{ext}'))
        
        if not documents_to_index:
            self.stdout.write(
                self.style.WARNING(f'No documents found in {kb_path}')
            )
            return
        
        self.stdout.write(f'Found {len(documents_to_index)} documents to index')
        
        # Index each document
        success_count = 0
        failed_count = 0
        total_chunks = 0
        
        for i, doc_path in enumerate(documents_to_index, 1):
            self.stdout.write(f'[{i}/{len(documents_to_index)}] Indexing {doc_path.name}...')
            
            try:
                # Create document record if not exists
                document, created = Document.objects.get_or_create(
                    file=str(doc_path.relative_to(settings.MEDIA_ROOT)),
                    defaults={
                        'title': doc_path.stem,
                        'category': 'other',
                        'file_type': doc_path.suffix[1:],
                        'file_size': doc_path.stat().st_size,
                        'status': 'processing'
                    }
                )
                
                # Index document
                chunk_count = indexer.index_document(
                    file_path=str(doc_path),
                    document_id=str(document.id),
                    document_title=document.title,
                    batch_size=options['batch_size']
                )
                
                # Update document status
                document.mark_as_indexed(chunk_count)
                
                success_count += 1
                total_chunks += chunk_count
                
                self.stdout.write(
                    self.style.SUCCESS(f'  ✓ Indexed {chunk_count} chunks')
                )
                
            except Exception as e:
                failed_count += 1
                logger.error(f'Failed to index {doc_path}: {str(e)}', exc_info=True)
                self.stdout.write(
                    self.style.ERROR(f'  ✗ Failed: {str(e)}')
                )
                
                # Mark as failed if document exists
                if document:
                    document.mark_as_failed()
        
        # Print summary
        self.stdout.write('\n' + '='*50)
        self.stdout.write(self.style.SUCCESS('Indexing Complete!'))
        self.stdout.write(f'  Successfully indexed: {success_count} documents')
        self.stdout.write(f'  Failed: {failed_count} documents')
        self.stdout.write(f'  Total chunks: {total_chunks}')
        self.stdout.write(f'  Vector store stats: {vector_store.get_stats()}')
        self.stdout.write('='*50)
    
    def _reindex_from_db(self, indexer, batch_size):
        """Reindex all documents from database"""
        documents = Document.objects.all()
        
        if not documents.exists():
            self.stdout.write(self.style.WARNING('No documents in database'))
            return
        
        self.stdout.write(f'Reindexing {documents.count()} documents...')
        
        success_count = 0
        failed_count = 0
        
        for i, document in enumerate(documents, 1):
            self.stdout.write(f'[{i}/{documents.count()}] Reindexing {document.title}...')
            
            try:
                document.status = 'processing'
                document.save()
                
                chunk_count = indexer.index_document(
                    file_path=document.file.path,
                    document_id=str(document.id),
                    document_title=document.title,
                    batch_size=batch_size
                )
                
                document.mark_as_indexed(chunk_count)
                success_count += 1
                
                self.stdout.write(
                    self.style.SUCCESS(f'  ✓ Reindexed {chunk_count} chunks')
                )
                
            except Exception as e:
                failed_count += 1
                logger.error(f'Failed to reindex {document.title}: {str(e)}')
                document.mark_as_failed()
                self.stdout.write(
                    self.style.ERROR(f'  ✗ Failed: {str(e)}')
                )
        
        self.stdout.write('\n' + '='*50)
        self.stdout.write(self.style.SUCCESS('Reindexing Complete!'))
        self.stdout.write(f'  Successfully reindexed: {success_count}')
        self.stdout.write(f'  Failed: {failed_count}')
        self.stdout.write('='*50)