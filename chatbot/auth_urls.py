"""
Authentication URLs
Add this to chatbot/auth_urls.py
"""

from django.urls import path
from . import auth_views
from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
    # Registration & Login
    # path('register/', auth_views.register_user, name='register'),
    # path('login/', auth_views.login_user, name='login'),
    # path('logout/', auth_views.logout_user, name='logout'),
    
    # # JWT Token endpoints
    # path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    
    # # User Profile
    # path('profile/', auth_views.get_user_profile, name='user_profile'),
    # path('profile/update/', auth_views.update_user_profile, name='update_profile'),
    
    # # Password Management
    # path('change-password/', auth_views.change_password, name='change_password'),
    
    # # Account Deletion
    # path('account/delete/', auth_views.delete_user_account, name='delete_account'),
]