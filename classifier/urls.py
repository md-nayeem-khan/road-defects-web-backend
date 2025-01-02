# classifier/urls.py
from django.urls import path
from .views import ImagePredictionView
from .views import HelloWorldView

urlpatterns = [
    path('predict/', ImagePredictionView.as_view(), name='image-predict'),
    path('hello/', HelloWorldView.as_view(), name='hello-world'),
]
