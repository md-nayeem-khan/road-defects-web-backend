from django.shortcuts import render

# classifier/views.py
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .model_loader import classifier  # Assuming you have a classifier class
from django.core.files.storage import default_storage
from django.conf import settings
import os

class HelloWorldView(APIView):
    def get(self, request, *args, **kwargs):
        return Response({"message": "Hello, World!"}, status=status.HTTP_200_OK)

class ImagePredictionView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        # Get the image from the request
        image_file = request.FILES.get('image')

        if image_file is None:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Save the image to the media directory (you can customize the path here)
        #image_path = os.path.join(settings.MEDIA_ROOT, image_file.name)
        #with default_storage.open(image_path, 'wb') as destination:
        #    for chunk in image_file.chunks():
        #        destination.write(chunk)
        
        # Read image bytes
        image_bytes = image_file.read()

        # Get prediction
        predicted_classes = classifier.predict(image_bytes)
        image_url = f"{settings.MEDIA_URL}{image_file.name}"
        predicted_classes_str = ", ".join(predicted_classes)
        # Return the prediction result
        response_data = {
            "image": image_url,  # URL to the uploaded image
            "result": predicted_classes_str  # Predicted class names
        }

        # Return the response
        return Response(response_data, status=status.HTTP_200_OK)
