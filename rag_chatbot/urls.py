"""
URL Configuration for the project
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework import permissions

# API Documentation
schema_view = get_schema_view(
    openapi.Info(
        title="Customer Support Chatbot API",
        default_version='v1',
        description="API for RAG-based customer support chatbot",
        terms_of_service="https://www.yourcompany.com/terms/",
        contact=openapi.Contact(email="support@yourcompany.com"),
        license=openapi.License(name="MIT License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    # Admin
    # path('admin/', admin.site.urls),
    
    # # Authentication endpoints
    # path('api/auth/', include('chatbot.auth_urls')),
    # # Authentication
    
    # path('api/auth/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    # path('api/auth/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    
    # API Documentation
    # path('api/docs/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    # path('api/redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    
    # Chatbot app
    path('', include('chatbot.urls')),
]

# Static/media are intentionally not served directly by Django.
