"""
Database models for the chatbot application
"""

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import uuid


class Document(models.Model):
    """Model for storing uploaded documents"""
    
    CATEGORY_CHOICES = [
        ('user_manual', 'User Manual'),
        ('faq', 'FAQ'),
        ('policy', 'Policy'),
        ('guide', 'Guide'),
        ('technical', 'Technical Documentation'),
        ('other', 'Other'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('indexed', 'Indexed'),
        ('failed', 'Failed'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='knowledge_base/')
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES, default='other')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    file_type = models.CharField(max_length=10)
    file_size = models.IntegerField(help_text="File size in bytes")
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    indexed_at = models.DateTimeField(null=True, blank=True)
    chunk_count = models.IntegerField(default=0, help_text="Number of chunks created")
    metadata = models.JSONField(default=dict, blank=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-uploaded_at']
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['category']),
            models.Index(fields=['is_active']),
        ]
    
    def __str__(self):
        return self.title
    
    def mark_as_indexed(self, chunk_count=0):
        """Mark document as successfully indexed"""
        self.status = 'indexed'
        self.indexed_at = timezone.now()
        self.chunk_count = chunk_count
        self.save()
    
    def mark_as_failed(self):
        """Mark document indexing as failed"""
        self.status = 'failed'
        self.save()



class Conversation(models.Model):
    """Model for storing chat conversations"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations', null=True,blank=True)       
    title = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['user', '-updated_at']),
        ]
    
    def __str__(self):
        return f"{self.title or 'Conversation'} - {self.user.username}"
    
    def get_message_count(self):
        """Get total number of messages in conversation"""
        return self.messages.count()
    
    def update_title_from_first_message(self):
        """Auto-generate title from first user message"""
        if not self.title:
            first_message = self.messages.filter(role='user').first()
            if first_message:
                self.title = first_message.content[:50] + ('...' if len(first_message.content) > 50 else '')
                self.save()


class Message(models.Model):
    """Model for storing individual messages"""
    
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    # RAG-specific fields
    retrieved_chunks = models.JSONField(default=list, blank=True, help_text="Retrieved context chunks")
    sources = models.JSONField(default=list, blank=True, help_text="Source documents used")
    token_count = models.IntegerField(null=True, blank=True)
    
    # Performance metrics
    response_time = models.FloatField(null=True, blank=True, help_text="Response time in seconds")
    
    # Feedback
    rating = models.IntegerField(null=True, blank=True, help_text="User rating (1-5)")
    feedback = models.TextField(blank=True)
    
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        ordering = ['created_at']
        indexes = [
            models.Index(fields=['conversation', 'created_at']),
            models.Index(fields=['role']),
        ]
    
    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."
    
    def add_rating(self, rating, feedback=''):
        """Add user rating and feedback"""
        self.rating = rating
        self.feedback = feedback
        self.save()


class DocumentChunk(models.Model):
    """Model for storing document chunks (optional - can also use FAISS only)"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks')
    chunk_index = models.IntegerField()
    content = models.TextField()
    embedding_id = models.CharField(max_length=255, help_text="ID in FAISS index")
    
    # Metadata
    page_number = models.IntegerField(null=True, blank=True)
    section = models.CharField(max_length=255, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['document', 'chunk_index']
        indexes = [
            models.Index(fields=['document', 'chunk_index']),
        ]
        unique_together = ['document', 'chunk_index']
    
    def __str__(self):
        return f"{self.document.title} - Chunk {self.chunk_index}"


class QueryLog(models.Model):
    """Model for logging queries for analytics"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    query = models.TextField()
    response = models.TextField()
    
    # Performance metrics
    retrieval_time = models.FloatField(help_text="Time for vector search in seconds")
    llm_time = models.FloatField(help_text="Time for LLM generation in seconds")
    total_time = models.FloatField(help_text="Total response time in seconds")
    
    # Retrieved context info
    chunks_retrieved = models.IntegerField(default=0)
    top_similarity_score = models.FloatField(null=True, blank=True)
    
    # User satisfaction
    was_helpful = models.BooleanField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['user']),
        ]
    
    def __str__(self):
        return f"Query: {self.query[:50]}..."


class RequestLog(models.Model):
    """Model for logging incoming requests"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    method = models.CharField(max_length=10)
    path = models.TextField()
    user_agent = models.TextField(blank=True)
    status_code = models.IntegerField()
    was_blocked = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['user']),
            models.Index(fields=['ip_address']),
        ]
    
    def __str__(self):
        return f"{self.method} {self.path} ({self.status_code})"


class SystemSettings(models.Model):
    """Model for storing system-wide settings"""
    
    key = models.CharField(max_length=100, unique=True, primary_key=True)
    value = models.JSONField()
    description = models.TextField(blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    updated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    
    class Meta:
        verbose_name = "System Setting"
        verbose_name_plural = "System Settings"
    
    def __str__(self):
        return self.key
    
    @classmethod
    def get_setting(cls, key, default=None):
        """Get a setting value"""
        try:
            return cls.objects.get(key=key).value
        except cls.DoesNotExist:
            return default
    
    @classmethod
    def set_setting(cls, key, value, description='', user=None):
        """Set a setting value"""
        setting, created = cls.objects.update_or_create(
            key=key,
            defaults={
                'value': value,
                'description': description,
                'updated_by': user,
            }
        )
        return setting
