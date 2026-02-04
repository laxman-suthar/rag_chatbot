"""
Serializers for the chatbot API
"""

from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Document, Conversation, Message, QueryLog


class UserSerializer(serializers.ModelSerializer):
    """Serializer for User model"""
    
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name']
        read_only_fields = ['id']


class DocumentSerializer(serializers.ModelSerializer):
    """Serializer for Document model"""
    
    uploaded_by = UserSerializer(read_only=True)
    file_url = serializers.SerializerMethodField()
    
    class Meta:
        model = Document
        fields = [
            'id', 'title', 'file', 'file_url', 'category', 'status',
            'file_type', 'file_size', 'uploaded_by', 'uploaded_at',
            'indexed_at', 'chunk_count', 'metadata', 'is_active'
        ]
        read_only_fields = [
            'id', 'status', 'file_type', 'file_size', 'uploaded_at',
            'indexed_at', 'chunk_count'
        ]
    
    def get_file_url(self, obj):
        """Get absolute URL for file"""
        request = self.context.get('request')
        if obj.file and request:
            return request.build_absolute_uri(obj.file.url)
        return None


class DocumentUploadSerializer(serializers.Serializer):
    """Serializer for document upload"""
    
    file = serializers.FileField()
    title = serializers.CharField(max_length=255, required=False)
    category = serializers.ChoiceField(
        choices=Document.CATEGORY_CHOICES,
        default='other'
    )
    
    def validate_file(self, value):
        """Validate file type and size"""
        from django.conf import settings
        
        # Check file extension
        ext = value.name.split('.')[-1].lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise serializers.ValidationError(
                f"File type .{ext} not allowed. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # Check file size
        if value.size > settings.MAX_UPLOAD_SIZE:
            raise serializers.ValidationError(
                f"File size exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE / 1024 / 1024}MB"
            )
        
        return value


class MessageSerializer(serializers.ModelSerializer):
    """Serializer for Message model"""
    
    class Meta:
        model = Message
        fields = [
            'id', 'conversation', 'role', 'content', 'created_at',
            'retrieved_chunks', 'sources', 'token_count',
            'response_time', 'rating', 'feedback', 'metadata'
        ]
        read_only_fields = [
            'id', 'created_at', 'retrieved_chunks', 'sources',
            'token_count', 'response_time'
        ]


class ConversationSerializer(serializers.ModelSerializer):
    """Serializer for Conversation model"""
    
    user = UserSerializer(read_only=True)
    message_count = serializers.SerializerMethodField()
    last_message = serializers.SerializerMethodField()
    
    class Meta:
        model = Conversation
        fields = [
            'id', 'user', 'title', 'created_at', 'updated_at',
            'is_active', 'metadata', 'message_count', 'last_message'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def get_message_count(self, obj):
        """Get total number of messages"""
        return obj.messages.count()
    
    def get_last_message(self, obj):
        """Get last message in conversation"""
        last_msg = obj.messages.last()
        if last_msg:
            return {
                'content': last_msg.content[:100],
                'role': last_msg.role,
                'created_at': last_msg.created_at
            }
        return None


class ConversationDetailSerializer(ConversationSerializer):
    """Detailed serializer with all messages"""
    
    messages = MessageSerializer(many=True, read_only=True)
    
    class Meta(ConversationSerializer.Meta):
        fields = ConversationSerializer.Meta.fields + ['messages']


class ChatRequestSerializer(serializers.Serializer):
    """Serializer for chat requests"""
    
    message = serializers.CharField(max_length=5000)
    conversation_id = serializers.UUIDField(required=False, allow_null=True)
    include_sources = serializers.BooleanField(default=True)
    stream = serializers.BooleanField(default=False)
    
    def validate_message(self, value):
        """Validate message is not empty"""
        if not value.strip():
            raise serializers.ValidationError("Message cannot be empty")
        return value.strip()


class ChatResponseSerializer(serializers.Serializer):
    """Serializer for chat responses"""
    
    message_id = serializers.UUIDField()
    conversation_id = serializers.UUIDField()
    response = serializers.CharField()
    sources = serializers.ListField(required=False)
    retrieved_chunks = serializers.ListField(required=False)
    response_time = serializers.FloatField(required=False)
    created_at = serializers.DateTimeField()


class MessageRatingSerializer(serializers.Serializer):
    """Serializer for rating messages"""
    
    rating = serializers.IntegerField(min_value=1, max_value=5)
    feedback = serializers.CharField(required=False, allow_blank=True, max_length=1000)
    
    def validate_rating(self, value):
        """Validate rating is between 1 and 5"""
        if value < 1 or value > 5:
            raise serializers.ValidationError("Rating must be between 1 and 5")
        return value


class QueryLogSerializer(serializers.ModelSerializer):
    """Serializer for QueryLog model"""
    
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = QueryLog
        fields = [
            'id', 'user', 'query', 'response', 'retrieval_time',
            'llm_time', 'total_time', 'chunks_retrieved',
            'top_similarity_score', 'was_helpful', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class AnalyticsSerializer(serializers.Serializer):
    """Serializer for analytics data"""
    
    total_queries = serializers.IntegerField()
    total_conversations = serializers.IntegerField()
    total_documents = serializers.IntegerField()
    avg_response_time = serializers.FloatField()
    avg_rating = serializers.FloatField()
    queries_per_day = serializers.ListField()
    top_queries = serializers.ListField()
    document_usage = serializers.ListField()


class BulkDocumentStatusSerializer(serializers.Serializer):
    """Serializer for bulk document operations"""
    
    document_ids = serializers.ListField(
        child=serializers.UUIDField(),
        min_length=1
    )
    action = serializers.ChoiceField(choices=['activate', 'deactivate', 'delete', 'reindex'])
    
    def validate_document_ids(self, value):
        """Validate all document IDs exist"""
        from .models import Document
        existing_ids = set(Document.objects.filter(id__in=value).values_list('id', flat=True))
        invalid_ids = set(value) - existing_ids
        if invalid_ids:
            raise serializers.ValidationError(
                f"Invalid document IDs: {', '.join(str(id) for id in invalid_ids)}"
            )
        return value