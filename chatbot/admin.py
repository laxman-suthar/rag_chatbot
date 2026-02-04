"""
Admin configuration for chatbot models
"""

from django.contrib import admin
from django.utils.html import format_html
from .models import Document, Conversation, Message, QueryLog, SystemSettings


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    """Admin for Document model"""
    
    list_display = [
        'title', 'category', 'status', 'file_type',
        'file_size_display', 'chunk_count', 'uploaded_at', 'is_active'
    ]
    list_filter = ['status', 'category', 'file_type', 'is_active', 'uploaded_at']
    search_fields = ['title', 'uploaded_by__username']
    readonly_fields = ['id', 'uploaded_at', 'indexed_at', 'chunk_count']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'title', 'file', 'category')
        }),
        ('Status', {
            'fields': ('status', 'is_active')
        }),
        ('Metadata', {
            'fields': ('file_type', 'file_size', 'chunk_count')
        }),
        ('Timestamps', {
            'fields': ('uploaded_at', 'indexed_at', 'uploaded_by')
        }),
        ('Additional Data', {
            'fields': ('metadata',),
            'classes': ('collapse',)
        }),
    )
    
    actions = ['activate_documents', 'deactivate_documents', 'reindex_documents']
    
    def file_size_display(self, obj):
        """Display file size in human-readable format"""
        size = obj.file_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    file_size_display.short_description = 'File Size'
    
    def activate_documents(self, request, queryset):
        """Activate selected documents"""
        updated = queryset.update(is_active=True)
        self.message_user(request, f'{updated} documents activated')
    activate_documents.short_description = 'Activate selected documents'
    
    def deactivate_documents(self, request, queryset):
        """Deactivate selected documents"""
        updated = queryset.update(is_active=False)
        self.message_user(request, f'{updated} documents deactivated')
    deactivate_documents.short_description = 'Deactivate selected documents'
    
    def reindex_documents(self, request, queryset):
        """Reindex selected documents"""
        for doc in queryset:
            doc.status = 'processing'
            doc.save()
            # In production, trigger Celery task here
        self.message_user(request, f'{queryset.count()} documents queued for reindexing')
    reindex_documents.short_description = 'Reindex selected documents'


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    """Admin for Conversation model"""
    
    list_display = ['id', 'user', 'title', 'message_count_display', 'created_at', 'is_active']
    list_filter = ['is_active', 'created_at']
    search_fields = ['user__username', 'title']
    readonly_fields = ['id', 'created_at', 'updated_at']
    
    def message_count_display(self, obj):
        """Display number of messages"""
        return obj.messages.count()
    message_count_display.short_description = 'Messages'


class MessageInline(admin.TabularInline):
    """Inline messages for conversation"""
    model = Message
    fields = ['role', 'content_preview', 'rating', 'created_at']
    readonly_fields = ['content_preview', 'created_at']
    extra = 0
    
    def content_preview(self, obj):
        """Show preview of message content"""
        return obj.content[:100] + '...' if len(obj.content) > 100 else obj.content
    content_preview.short_description = 'Content'


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    """Admin for Message model"""
    
    list_display = [
        'id', 'conversation', 'role', 'content_preview',
        'rating', 'response_time', 'created_at'
    ]
    list_filter = ['role', 'rating', 'created_at']
    search_fields = ['conversation__user__username', 'content']
    readonly_fields = ['id', 'created_at', 'retrieved_chunks', 'sources']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'conversation', 'role', 'content', 'created_at')
        }),
        ('RAG Data', {
            'fields': ('retrieved_chunks', 'sources', 'token_count', 'response_time'),
            'classes': ('collapse',)
        }),
        ('Feedback', {
            'fields': ('rating', 'feedback')
        }),
        ('Metadata', {
            'fields': ('metadata',),
            'classes': ('collapse',)
        }),
    )
    
    def content_preview(self, obj):
        """Show preview of content"""
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content
    content_preview.short_description = 'Content'


@admin.register(QueryLog)
class QueryLogAdmin(admin.ModelAdmin):
    """Admin for QueryLog model"""
    
    list_display = [
        'id', 'user', 'query_preview', 'total_time',
        'chunks_retrieved', 'was_helpful', 'created_at'
    ]
    list_filter = ['was_helpful', 'created_at']
    search_fields = ['user__username', 'query', 'response']
    readonly_fields = [
        'id', 'user', 'query', 'response', 'retrieval_time',
        'llm_time', 'total_time', 'chunks_retrieved',
        'top_similarity_score', 'created_at'
    ]
    
    def query_preview(self, obj):
        """Show preview of query"""
        return obj.query[:50] + '...' if len(obj.query) > 50 else obj.query
    query_preview.short_description = 'Query'
    
    def has_add_permission(self, request):
        """Don't allow manual addition"""
        return False


@admin.register(SystemSettings)
class SystemSettingsAdmin(admin.ModelAdmin):
    """Admin for SystemSettings model"""
    
    list_display = ['key', 'description', 'updated_at', 'updated_by']
    search_fields = ['key', 'description']
    readonly_fields = ['updated_at']
    
    fieldsets = (
        (None, {
            'fields': ('key', 'value', 'description')
        }),
        ('Metadata', {
            'fields': ('updated_at', 'updated_by'),
            'classes': ('collapse',)
        }),
    )


# Customize admin site
admin.site.site_header = "Customer Support Chatbot Admin"
admin.site.site_title = "Chatbot Admin"
admin.site.index_title = "Welcome to Chatbot Administration"