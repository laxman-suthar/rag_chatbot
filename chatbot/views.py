"""
Views for the chatbot application
"""

import logging
import time
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.db import transaction
from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
from django.db.models import Count, Avg, Q
from django.utils import timezone
from datetime import timedelta

from .models import Document, Conversation, Message, QueryLog
from .serializers import (
    DocumentSerializer, DocumentUploadSerializer,
    ConversationSerializer, ConversationDetailSerializer,
    MessageSerializer, ChatRequestSerializer, ChatResponseSerializer,
    MessageRatingSerializer, QueryLogSerializer, AnalyticsSerializer
)

# Import RAG components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'rag_service'))

from rag_service.rag_engine import RAGEngine
from rag_service.embeddings import EmbeddingModel
from rag_service.vector_store import VectorStore
from rag_service.document_processor import DocumentIndexer

logger = logging.getLogger('chatbot')

# Initialize RAG engine (singleton)
rag_engine = None

def get_rag_engine():
    """Get or create RAG engine instance"""
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine()
    return rag_engine


# Template Views
@login_required
def index(request):
    """Main chat interface"""
    return render(request, 'chatbot/index.html')


@login_required
def chat_view(request):
    """Chat page with conversation history"""
    conversations = Conversation.objects.filter(
        user=request.user,
        is_active=True
    ).order_by('-updated_at')[:10]
    
    return render(request, 'chatbot/chat.html', {
        'conversations': conversations
    })


# API ViewSets
class DocumentViewSet(viewsets.ModelViewSet):
    """ViewSet for Document model"""
    
    serializer_class = DocumentSerializer
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    
    def get_queryset(self):
        """Get documents for current user or all if admin"""
        if self.request.user.is_staff:
            return Document.objects.all()
        return Document.objects.filter(uploaded_by=self.request.user)
    
    @action(detail=False, methods=['post'])
    def upload(self, request):
        """Upload and index a new document"""
        serializer = DocumentUploadSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            file = serializer.validated_data['file']
            title = serializer.validated_data.get('title', file.name)
            category = serializer.validated_data.get('category', 'other')
            
            # Create document record
            document = Document.objects.create(
                title=title,
                file=file,
                category=category,
                file_type=file.name.split('.')[-1].lower(),
                file_size=file.size,
                uploaded_by=request.user,
                status='processing'
            )
            
            # Start indexing (in production, use Celery)
            try:
                self._index_document(document)
                document.mark_as_indexed()
            except Exception as e:
                logger.error(f"Failed to index document: {str(e)}")
                document.mark_as_failed()
                raise
            
            return Response(
                DocumentSerializer(document).data,
                status=status.HTTP_201_CREATED
            )
            
        except Exception as e:
            logger.error(f"Error uploading document: {str(e)}", exc_info=True)
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _index_document(self, document):
        """Index a document into vector store"""
        embedding_model = EmbeddingModel()
        vector_store = VectorStore()
        indexer = DocumentIndexer(embedding_model, vector_store)
        
        chunk_count = indexer.index_document(
            file_path=document.file.path,
            document_id=str(document.id),
            document_title=document.title
        )
        
        document.chunk_count = chunk_count
        document.save()
        
        logger.info(f"Indexed document {document.title} with {chunk_count} chunks")
    
    @action(detail=True, methods=['post'])
    def reindex(self, request, pk=None):
        """Reindex a document"""
        document = self.get_object()
        
        try:
            document.status = 'processing'
            document.save()
            
            self._index_document(document)
            document.mark_as_indexed()
            
            return Response({'message': 'Document reindexed successfully'})
            
        except Exception as e:
            document.mark_as_failed()
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ConversationViewSet(viewsets.ModelViewSet):
    """ViewSet for Conversation model"""
    
    serializer_class = ConversationSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Get conversations for current user"""
        return Conversation.objects.filter(
            user=self.request.user
        ).prefetch_related('messages')
    
    def get_serializer_class(self):
        """Use detailed serializer for retrieve action"""
        if self.action == 'retrieve':
            return ConversationDetailSerializer
        return ConversationSerializer
    
    def perform_create(self, serializer):
        """Set user when creating conversation"""
        serializer.save(user=self.request.user)
    
    @action(detail=True, methods=['delete'])
    def archive(self, request, pk=None):
        """Archive a conversation"""
        conversation = self.get_object()
        conversation.is_active = False
        conversation.save()
        
        return Response({'message': 'Conversation archived'})


class MessageViewSet(viewsets.ModelViewSet):
    """ViewSet for Message model"""
    
    serializer_class = MessageSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Get messages for current user's conversations"""
        return Message.objects.filter(
            conversation__user=self.request.user
        )
    
    @action(detail=True, methods=['post'])
    def rate(self, request, pk=None):
        """Rate a message"""
        message = self.get_object()
        serializer = MessageRatingSerializer(data=request.data)
        
        if serializer.is_valid():
            message.add_rating(
                rating=serializer.validated_data['rating'],
                feedback=serializer.validated_data.get('feedback', '')
            )
            return Response({'message': 'Rating saved'})
        
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def chat_api(request):
    """
    Main chat endpoint for sending messages and getting responses
    """
    serializer = ChatRequestSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        message_text = serializer.validated_data['message']
        conversation_id = serializer.validated_data.get('conversation_id')
        include_sources = serializer.validated_data.get('include_sources', True)
        
        # Get or create conversation
        if conversation_id:
            conversation = Conversation.objects.get(
                id=conversation_id,
                user=request.user
            )
        else:
            conversation = Conversation.objects.create(
                user=request.user
            )
        
        # Create user message
        user_message = Message.objects.create(
            conversation=conversation,
            role='user',
            content=message_text
        )
        
        # Get conversation history
        history_messages = Message.objects.filter(
            conversation=conversation
        ).order_by('created_at')[:10]  # Last 10 messages
        
        conversation_history = [
            {
                'role': msg.role,
                'content': msg.content
            }
            for msg in history_messages
        ]
        
        # Get response from RAG engine
        start_time = time.time()
        rag = get_rag_engine()
        rag_response = rag.get_response(
            query=message_text,
            conversation_history=conversation_history[:-1]  # Exclude current message
        )
        
        # Create assistant message
        assistant_message = Message.objects.create(
            conversation=conversation,
            role='assistant',
            content=rag_response['response'],
            retrieved_chunks=rag_response.get('retrieved_chunks', []),
            sources=rag_response.get('sources', []),
            response_time=rag_response['total_time']
        )
        
        # Log query for analytics
        QueryLog.objects.create(
            user=request.user,
            query=message_text,
            response=rag_response['response'],
            retrieval_time=rag_response.get('retrieval_time', 0),
            llm_time=rag_response.get('llm_time', 0),
            total_time=rag_response['total_time'],
            chunks_retrieved=len(rag_response.get('retrieved_chunks', [])),
            top_similarity_score=rag_response['retrieved_chunks'][0]['score'] if rag_response.get('retrieved_chunks') else None
        )
        
        # Update conversation title if first message
        if conversation.messages.count() == 2:  # User + Assistant
            conversation.update_title_from_first_message()
        
        # Prepare response
        response_data = {
            'message_id': str(assistant_message.id),
            'conversation_id': str(conversation.id),
            'response': rag_response['response'],
            'response_time': rag_response['total_time'],
            'created_at': assistant_message.created_at
        }
        
        if include_sources:
            response_data['sources'] = rag_response.get('sources', [])
        
        return Response(response_data)
        
    except Conversation.DoesNotExist:
        return Response(
            {'error': 'Conversation not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        logger.error(f"Error in chat_api: {str(e)}", exc_info=True)
        return Response(
            {'error': 'An error occurred processing your request'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def analytics_api(request):
    """Get analytics data"""
    try:
        # Date range
        days = int(request.GET.get('days', 30))
        start_date = timezone.now() - timedelta(days=days)
        
        # Total queries
        total_queries = QueryLog.objects.filter(
            created_at__gte=start_date
        ).count()
        
        # Total conversations
        total_conversations = Conversation.objects.filter(
            created_at__gte=start_date
        ).count()
        
        # Total documents
        total_documents = Document.objects.filter(
            status='indexed'
        ).count()
        
        # Average response time
        avg_response_time = QueryLog.objects.filter(
            created_at__gte=start_date
        ).aggregate(Avg('total_time'))['total_time__avg'] or 0
        
        # Average rating
        avg_rating = Message.objects.filter(
            rating__isnull=False,
            created_at__gte=start_date
        ).aggregate(Avg('rating'))['rating__avg'] or 0
        
        # Queries per day
        queries_per_day = []
        for i in range(days):
            day = timezone.now() - timedelta(days=i)
            count = QueryLog.objects.filter(
                created_at__date=day.date()
            ).count()
            queries_per_day.append({
                'date': day.date().isoformat(),
                'count': count
            })
        
        # Top queries
        top_queries = QueryLog.objects.filter(
            created_at__gte=start_date
        ).values('query').annotate(
            count=Count('query')
        ).order_by('-count')[:10]
        
        analytics_data = {
            'total_queries': total_queries,
            'total_conversations': total_conversations,
            'total_documents': total_documents,
            'avg_response_time': round(avg_response_time, 2),
            'avg_rating': round(avg_rating, 2),
            'queries_per_day': queries_per_day,
            'top_queries': list(top_queries),
            'document_usage': []  # Can be implemented later
        }
        
        serializer = AnalyticsSerializer(analytics_data)
        return Response(serializer.data)
        
    except Exception as e:
        logger.error(f"Error in analytics_api: {str(e)}", exc_info=True)
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """Health check endpoint"""
    try:
        # Check database
        Document.objects.count()
        
        # Check vector store
        rag = get_rag_engine()
        stats = rag.vector_store.get_stats()
        
        return Response({
            'status': 'healthy',
            'database': 'ok',
            'vector_store': 'ok',
            'indexed_documents': stats['total_vectors']
        })
    except Exception as e:
        return Response({
            'status': 'unhealthy',
            'error': str(e)
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)