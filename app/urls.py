from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

# urlpatterns = [
#     # path('', views.home, name='home'),
#     path('upload/', views.upload_video, name='upload_video'),
# ]

# urls.py

# from django.urls import path
# from . import views

# urlpatterns = [
#     path('pose_detection/', views.pose_detection_view, name='pose_detection'),
# ]

urlpatterns = [
    path('', views.upload_video, name='upload_video'),
    path('process/', views.process_video, name='process_video'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)