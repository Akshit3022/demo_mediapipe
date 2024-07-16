from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('detect-landmarks/', views.detect_landmarks, name='detect_landmarks'),
    path('video_feed', views.video_feed, name='video_feed'),
]