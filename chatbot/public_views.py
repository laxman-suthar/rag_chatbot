"""
Public chat views for portfolio chatbot (no authentication required)
Add this to chatbot/public_views.py
"""

import logging
import time
import random
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework import status
from django.db import transaction

from .models import Conversation, Message, QueryLog
from .serializers import ChatRequestSerializer

# Import RAG components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'rag_service'))

from rag_service.rag_engine import RAGEngine

logger = logging.getLogger('chatbot')

# Initialize RAG engine (singleton)
rag_engine = None
GREETING_RESPONSES = [
    "Hi! How can I help you today?",
    "Hello! Ask me anything about Laxman's work or experience.",
    "Hey there! What would you like to know?",
]

def get_rag_engine():
    """Get or create RAG engine instance"""
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine()
    return rag_engine


def is_greeting(text):
    normalized = text.lower().strip()
    greetings = {
        "hi", "hello", "hey", "hii", "hiii", "hola", "namaste", "yo", "sup",
        "good morning", "good afternoon", "good evening"
    }
    return normalized in greetings


def get_greeting_response():
    return random.choice(GREETING_RESPONSES)


def portfolio_chatbot_view(request):
    """
    Public portfolio chatbot page (no login required)
    """
    return render(request, 'portfolio_chatbot.html')


@api_view(['POST'])
@permission_classes([AllowAny])  # No authentication required
@csrf_exempt
def public_chat_api(request):
    """
    Public chat endpoint for portfolio chatbot (no authentication required)
    
    POST /api/public/chat/
    {
        "message": "What are your skills?",
        "conversation_id": null
    }
    """
    try:
        message_text = request.data.get('message', '').strip()
        conversation_id = request.data.get('conversation_id')
        
        if not message_text:
            return Response(
                {'error': 'Message cannot be empty'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Get or create anonymous conversation
        if conversation_id:
            try:
                conversation = Conversation.objects.get(
                    id=conversation_id,
                    user=None  # Anonymous conversations have no user
                )
            except Conversation.DoesNotExist:
                conversation = Conversation.objects.create(
                    user=None,
                    title=message_text[:50]
                )
        else:
            conversation = Conversation.objects.create(
                user=None,
                title=message_text[:50]
            )

        # Create user message
        user_message = Message.objects.create(
            conversation=conversation,
            role='user',
            content=message_text
        )

        # Handle greetings without calling RAG
        if is_greeting(message_text):
            greeting_response = get_greeting_response()
            assistant_message = Message.objects.create(
                conversation=conversation,
                role='assistant',
                content=greeting_response,
                response_time=0
            )

            response_data = {
                'message_id': str(assistant_message.id),
                'conversation_id': str(conversation.id),
                'response': greeting_response,
                'response_time': 0,
                'created_at': assistant_message.created_at,
                'sources': []
            }
            return Response(response_data)
        
        # Get conversation history
        history_messages = Message.objects.filter(
            conversation=conversation
        ).order_by('created_at')[:10]
        
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
            conversation_history=conversation_history[:-1]
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
        
        # Log query (optional, for analytics)
        QueryLog.objects.create(
            user=None,
            query=message_text,
            response=rag_response['response'],
            retrieval_time=rag_response.get('retrieval_time', 0),
            llm_time=rag_response.get('llm_time', 0),
            total_time=rag_response['total_time'],
            chunks_retrieved=len(rag_response.get('retrieved_chunks', [])),
            top_similarity_score=rag_response['retrieved_chunks'][0]['score'] if rag_response.get('retrieved_chunks') else None
        )
        
        # Update conversation title if first message
        if conversation.messages.count() == 2:
            conversation.update_title_from_first_message()
        
        # Prepare response
        response_data = {
            'message_id': str(assistant_message.id),
            'conversation_id': str(conversation.id),
            'response': rag_response['response'],
            'response_time': rag_response['total_time'],
            'created_at': assistant_message.created_at,
            'sources': rag_response.get('sources', [])
        }
        
        return Response(response_data)
        
    except Exception as e:
        logger.error(f"Error in public_chat_api: {str(e)}", exc_info=True)
        return Response(
            {'error': 'An error occurred processing your request'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([AllowAny])
def public_health_check(request):
    """Health check for public endpoint"""
    try:
        from .models import Document
        doc_count = Document.objects.filter(status='indexed').count()
        
        return Response({
            'status': 'healthy',
            'indexed_documents': doc_count,
            'message': 'Portfolio chatbot is ready'
        })
    except Exception as e:
        return Response({
            'status': 'unhealthy',
            'error': str(e)
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
