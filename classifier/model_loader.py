import torch
from torchvision.models import resnet50
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

class ImageClassifier:
    def __init__(self, model_path, num_classes):
        # Define the model architecture for multi-label classification
        self.model = resnet50(pretrained=False)

        # Freeze feature extractor layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the fully connected layer to match the number of classes
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid()  # Sigmoid for multi-label classification
        )

        # Load the model weights
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()  # Set to evaluation mode

        # Define the image transformation for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_bytes):
        # Convert the byte stream to a PIL image
        image = Image.open(io.BytesIO(image_bytes))

        # Apply the transformations
        image = self.transform(image).unsqueeze(0)

        # Make the prediction
        with torch.no_grad():
            outputs = self.model(image)
            # Get the predicted labels (multi-label, output probabilities)
            predictions = (outputs > 0.5).float().cpu().numpy().flatten()  # Threshold at 0.5 for binary classification
        class_names = ['No defects', 'Bleeding', 'Connected crack', 'Corrugation shoving', 'Depression rutting', 'Isolated crack', 'Manhole', 'Patch', 'Pothole', 'Raveling', 'Sidewalk shoulder']
        predicted_classes = [class_names[i] for i, pred in enumerate(predictions) if pred == 1]    
        return predicted_classes  # Return a list of predicted labels

# Instantiate with the model path and the number of classes
num_classes = 11  # Change this to match the number of unique labels in your dataset
classifier = ImageClassifier(model_path='model.pth', num_classes=num_classes)
