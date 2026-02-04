"""
URL Configuration for chatbot app
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from . import public_views
# Create router for viewsets
router = DefaultRouter()
router.register(r'documents', views.DocumentViewSet, basename='document')
router.register(r'conversations', views.ConversationViewSet, basename='conversation')
router.register(r'messages', views.MessageViewSet, basename='message')

app_name = 'chatbot'

urlpatterns = [
    # Template views
    path('', public_views.portfolio_chatbot_view, name='portfolio_home'),
    path('', views.index, name='index'),
    path('chat/', views.chat_view, name='chat'),
    path('api/public/chat/', public_views.public_chat_api, name='public_chat_api'),
    # path('api/public/health/', public_views.public_health_check, name='public_health'),
    # API endpoints
    # path('api/', include(router.urls)),
    # path('api/chat/', views.chat_api, name='chat_api'),
    # path('api/analytics/', views.analytics_api, name='analytics_api'),
    # path('api/health/', views.health_check, name='health_check'),
]